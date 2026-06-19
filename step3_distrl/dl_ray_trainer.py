"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import shutil
import statistics
import uuid
from copy import deepcopy
from pprint import pprint
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
import json

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import simple_timer
from verl.workers.rollout.llm_server import LLMServerManager

from datetime import datetime
import pytz
shanghai_tz = pytz.timezone('Asia/Shanghai')
from .dl_core_algos import (
    build_candidate_token_mask_from_batch,
    compute_distrl_advantages_and_returns,
    compute_logprob_advantage_correlation,
)
TOPK = 5
MAX_SAMPLES_PER_BATCH = 1
DISTRL_ACTOR_ONLY_CONFIG_KEYS = {
    "pg_loss_coef",
    "distrl_loss_coef",
    "nll_loss_coef",
    "distrl_topk",
    "kl_threshold",
}


def require_reward_fn_key(batch: DataProto, reward_fn_key: str = "data_source") -> DataProto:
    if reward_fn_key not in batch.non_tensor_batch:
        raise KeyError(
            f"Missing batch.non_tensor_batch['{reward_fn_key}']; step3_distrl no longer infers reward keys implicitly."
        )
    return batch


class RewardFnWithRequiredKey:
    def __init__(self, reward_fn, reward_fn_key: str = "data_source"):
        self._reward_fn = reward_fn
        self.reward_fn_key = reward_fn_key

    def __call__(self, batch: DataProto, *args, **kwargs):
        batch = require_reward_fn_key(batch, reward_fn_key=self.reward_fn_key)
        return self._reward_fn(batch, *args, **kwargs)

    def verify(self, batch: DataProto, *args, **kwargs):
        batch = require_reward_fn_key(batch, reward_fn_key=self.reward_fn_key)
        return self._reward_fn.verify(batch, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._reward_fn, name)


def build_prompt_group_acc_tensor(scores, prompt_group_id: torch.Tensor, sample_rank_in_group: torch.Tensor):
    prompt_group_id = prompt_group_id.view(-1)
    sample_rank_in_group = sample_rank_in_group.view(-1)
    acc = torch.as_tensor(scores, device=prompt_group_id.device, dtype=torch.float32).view(-1)

    if not (acc.shape == prompt_group_id.shape == sample_rank_in_group.shape):
        raise ValueError(
            "scores, prompt_group_id, and sample_rank_in_group must have the same flattened shape. "
            f"Got {tuple(acc.shape)}, {tuple(prompt_group_id.shape)}, and {tuple(sample_rank_in_group.shape)}."
        )
    return acc


def build_prompt_group_weight_margin_tensors(
    acc: torch.Tensor,
    prompt_group_id: torch.Tensor,
    sample_rank_in_group: torch.Tensor,
    n_samples: int,
    base_margin: float,
    use_dlw: bool,
    use_adb: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    prompt_group_id = prompt_group_id.view(-1)
    sample_rank_in_group = sample_rank_in_group.view(-1)
    acc = torch.as_tensor(acc, device=prompt_group_id.device, dtype=torch.float32).view(-1)

    if not (acc.shape == prompt_group_id.shape == sample_rank_in_group.shape):
        raise ValueError(
            "acc, prompt_group_id, and sample_rank_in_group must have the same flattened shape. "
            f"Got {tuple(acc.shape)}, {tuple(prompt_group_id.shape)}, and {tuple(sample_rank_in_group.shape)}."
        )
    if n_samples <= 0:
        raise ValueError(f"rollout.n must be positive, got {n_samples}.")
    if acc.numel() % n_samples != 0:
        raise ValueError(f"Batch size {acc.numel()} must be divisible by rollout.n={n_samples}.")

    dlw_weight = torch.ones_like(acc)
    margin = torch.full_like(acc, float(base_margin))
    expected_ranks = torch.arange(n_samples, device=sample_rank_in_group.device, dtype=sample_rank_in_group.dtype)

    mixed_prompt_count = 0
    all_correct_prompt_count = 0
    all_wrong_prompt_count = 0
    max_dlw_norm_error = 0.0

    unique_prompt_ids = torch.unique(prompt_group_id, sorted=True)
    for prompt_id in unique_prompt_ids:
        group_indices = torch.nonzero(prompt_group_id == prompt_id, as_tuple=False).flatten()
        if group_indices.numel() != n_samples:
            raise ValueError(
                f"Prompt group {prompt_id.item()} has {group_indices.numel()} samples, expected rollout.n={n_samples}."
            )

        group_indices = group_indices[torch.argsort(sample_rank_in_group[group_indices])]
        group_ranks = sample_rank_in_group[group_indices]
        if not torch.equal(group_ranks, expected_ranks):
            raise ValueError(
                f"Prompt group {prompt_id.item()} has sample ranks {group_ranks.tolist()}, "
                f"expected {expected_ranks.tolist()}."
            )

        group_acc = acc[group_indices]
        correct_mask = group_acc > 0.5
        if torch.all(correct_mask):
            all_correct_prompt_count += 1
            dlw_weight[group_indices] = 0.0
            continue
        if not torch.any(correct_mask):
            all_wrong_prompt_count += 1
            dlw_weight[group_indices] = 0.0
            continue

        mixed_prompt_count += 1
        group_acc_mean = group_acc.mean()

        if use_dlw:
            group_weight = torch.where(correct_mask, 1.0 - group_acc_mean, group_acc_mean)
            group_weight_sum = group_weight.sum()
            if group_weight_sum.abs().item() < 1e-8:
                group_weight = torch.zeros_like(group_weight)
            else:
                group_weight = group_weight * (float(group_indices.numel()) / group_weight_sum)
        else:
            group_weight = torch.ones_like(group_acc)
        dlw_weight[group_indices] = group_weight
        max_dlw_norm_error = max(max_dlw_norm_error, abs(group_weight.sum().item() - float(group_indices.numel())))

        if use_adb:
            probability = torch.clamp(group_acc_mean, 1e-6, 1 - 1e-6)
            difficulty_boundary = torch.log(probability / (1 - probability))
            margin[group_indices] = torch.where(
                correct_mask,
                float(base_margin) - difficulty_boundary,
                float(base_margin) + difficulty_boundary,
            )

    metrics = {
        "reward_model/dlw_mixed_prompt_count": float(mixed_prompt_count),
        "reward_model/dlw_all_correct_prompt_count": float(all_correct_prompt_count),
        "reward_model/dlw_all_wrong_prompt_count": float(all_wrong_prompt_count),
        "reward_model/dlw_max_norm_error": float(max_dlw_norm_error),
    }
    return {"dlw_weight": dlw_weight, "margin": margin}, metrics


def build_prompt_group_chosen_rejected_tensors(
    acc: torch.Tensor,
    prompt_group_id: torch.Tensor,
    sample_rank_in_group: torch.Tensor,
    n_samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    prompt_group_id = prompt_group_id.view(-1)
    sample_rank_in_group = sample_rank_in_group.view(-1)
    acc = torch.as_tensor(acc, device=prompt_group_id.device, dtype=torch.float32).view(-1)

    if not (acc.shape == prompt_group_id.shape == sample_rank_in_group.shape):
        raise ValueError(
            "acc, prompt_group_id, and sample_rank_in_group must have the same flattened shape. "
            f"Got {tuple(acc.shape)}, {tuple(prompt_group_id.shape)}, and {tuple(sample_rank_in_group.shape)}."
        )
    if n_samples <= 0:
        raise ValueError(f"rollout.n must be positive, got {n_samples}.")
    if acc.numel() % n_samples != 0:
        raise ValueError(f"Batch size {acc.numel()} must be divisible by rollout.n={n_samples}.")

    chosen = torch.zeros_like(acc, dtype=torch.bool)
    rejected = torch.zeros_like(acc, dtype=torch.bool)
    expected_ranks = torch.arange(n_samples, device=sample_rank_in_group.device, dtype=sample_rank_in_group.dtype)

    valid_prompt_count = 0
    skipped_prompt_count = 0
    pair_count = 0

    unique_prompt_ids = torch.unique(prompt_group_id, sorted=True)
    for prompt_id in unique_prompt_ids:
        group_indices = torch.nonzero(prompt_group_id == prompt_id, as_tuple=False).flatten()
        if group_indices.numel() != n_samples:
            raise ValueError(
                f"Prompt group {prompt_id.item()} has {group_indices.numel()} samples, expected rollout.n={n_samples}."
            )

        group_indices = group_indices[torch.argsort(sample_rank_in_group[group_indices])]
        group_ranks = sample_rank_in_group[group_indices]
        if not torch.equal(group_ranks, expected_ranks):
            raise ValueError(
                f"Prompt group {prompt_id.item()} has sample ranks {group_ranks.tolist()}, "
                f"expected {expected_ranks.tolist()}."
            )

        group_acc = acc[group_indices]
        positive_indices = group_indices[group_acc > 0.5]
        negative_indices = group_indices[group_acc < 0.5]
        cur_pair_count = min(positive_indices.numel(), negative_indices.numel())
        if cur_pair_count == 0:
            skipped_prompt_count += 1
            continue

        valid_prompt_count += 1
        pair_count += int(cur_pair_count)
        chosen[positive_indices[:cur_pair_count]] = True
        rejected[negative_indices[:cur_pair_count]] = True

    metrics = {
        "reward_model/dpo_pair_count_precomputed": float(pair_count),
        "reward_model/dpo_valid_prompts_precomputed": float(valid_prompt_count),
        "reward_model/dpo_skipped_prompts_precomputed": float(skipped_prompt_count),
    }
    return {"chosen": chosen, "rejected": rejected}, metrics

def compute_advantage(data: DataProto, config):
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    advantages, normalized_candidate_td_advantages, advantage_metrics = compute_distrl_advantages_and_returns(
        data, 
        response_mask,
        config.actor_rollout_ref.rollout.n,
        config
    )
    data.batch['advantages'] = advantages
    data.batch["normalized_candidate_td_advantages"] = normalized_candidate_td_advantages
    return data, advantage_metrics


def compute_data_metrics(batch, use_critic=True):
    advantages = batch.batch["advantages"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    has_returns = "returns" in batch.batch.keys()

    if has_returns:
        returns = batch.batch["returns"]
        valid_returns = torch.masked_select(returns, response_mask)

    if use_critic and has_returns and "values" in batch.batch.keys():
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # breakpoint()
    metrics = {
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        **(
            {
                "critic/returns/mean": torch.mean(valid_returns).detach().item(),
                "critic/returns/max": torch.max(valid_returns).detach().item(),
                "critic/returns/min": torch.min(valid_returns).detach().item(),
            }
            if has_returns
            else {}
        ),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic and has_returns and "values" in batch.batch.keys()
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


class RayDistRLTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        device_name="cuda",
    ):
        # assert get_torch_device().is_available(), 'cuda must be available on driver'

        reward_fn_key = config.data.get("reward_fn_key", "data_source")
        if reward_fn is not None:
            reward_fn = RewardFnWithRequiredKey(reward_fn, reward_fn_key=reward_fn_key)
        if val_reward_fn is not None:
            val_reward_fn = RewardFnWithRequiredKey(val_reward_fn, reward_fn_key=reward_fn_key)

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            device_name=device_name,
        )

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.use_critic = False

    def _validate_config(self):
        if hasattr(super(), "_validate_config"):
            super()._validate_config()
        # TODO: Additional config checks can be added here

    def _actor_rollout_worker_config(self):
        worker_config = deepcopy(self.config.actor_rollout_ref)
        with open_dict(worker_config.actor):
            for key in DISTRL_ACTOR_ONLY_CONFIG_KEYS:
                worker_config.actor.pop(key, None)
        return worker_config

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        actor_rollout_worker_config = self._actor_rollout_worker_config()

        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(Role.ActorRollout)] = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=actor_rollout_worker_config,
            distillation_config=self.config.get("distillation"),
            role=str(Role.ActorRollout),
        )

        actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        self.resource_pool_to_cls[actor_resource_pool][str(Role.Actor)] = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role=str(Role.Actor),
        )

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            ref_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            self.resource_pool_to_cls[ref_resource_pool][str(Role.RefPolicy)] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )

        if self.use_rm and Role.RewardModel in self.role_worker_mapping:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            self.resource_pool_to_cls[rm_resource_pool][str(Role.RewardModel)] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        self.actor_train_wg = all_wg[str(Role.Actor)]
        self.actor_train_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=None,
            reward_loop_worker_handles=None,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        self.checkpoint_manager.sleep_replicas()

    def _sync_dist_actor_to_vllm_worker(self, global_step: int):
        sync_dir = os.path.join(self.config.trainer.default_local_dir, "_distrl_vllm_sync", f"global_step_{global_step}")
        actor_path = os.path.join(sync_dir, "actor")
        if os.path.exists(sync_dir):
            shutil.rmtree(sync_dir)
        os.makedirs(sync_dir, exist_ok=True)
        self.actor_train_wg.save_checkpoint(actor_path, None, global_step, max_ckpt_to_keep=1)
        self.actor_rollout_wg.load_checkpoint(actor_path, None, del_local_after_load=False)

    def _create_dataloader(self, *args, **kwargs):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            data_files=self.config.data.train_files, tokenizer=self.tokenizer, config=self.config.data
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed") or 1)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=int(self.config.data.train_batch_size * self.config.data.oversample_factor),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            data_files=self.config.data.val_files, tokenizer=self.tokenizer, config=self.config.data
        )
        val_batch_size = self.config.data.get("val_batch_size", len(self.val_dataset))
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        val_batch_size = max(1, min(int(val_batch_size), len(self.val_dataset)))
        val_repeat_times = max(1, int(self.config.actor_rollout_ref.rollout.val_kwargs.n))
        max_num_seqs = self.config.actor_rollout_ref.rollout.get("max_num_seqs", None)
        if max_num_seqs is not None:
            val_batch_size = min(val_batch_size, max(1, int(max_num_seqs) // val_repeat_times))
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            shuffle=bool(self.config.data.get("validation_shuffle", False)),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if OmegaConf.select(self.config, "critic.optim"):
                self.config.critic.optim.total_training_steps = total_training_steps

    def _drop_overlong_generation_prompts(self, batch: DataProto, context: str) -> tuple[DataProto | None, int]:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            return batch, 0

        prompt_budget = int(
            self.config.actor_rollout_ref.rollout.prompt_length + self.config.actor_rollout_ref.rollout.response_length
        )
        valid_indices = []
        for idx, raw_prompt in enumerate(raw_prompts):
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    raw_prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                prompt_len = len(prompt_ids)
            except Exception:
                prompt_len = int(batch.batch["attention_mask"][idx, : batch.batch["prompts"].shape[-1]].sum().item())
            if prompt_len < prompt_budget:
                valid_indices.append(idx)

        skipped = len(raw_prompts) - len(valid_indices)
        if skipped > 0:
            print(f"Skipped {skipped} {context} samples with no vLLM generation budget.")
        if not valid_indices:
            return None, skipped
        if skipped > 0:
            batch.reorder(torch.tensor(valid_indices, dtype=torch.long))
        return batch, skipped

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict = {"acc": []}
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        skipped_validation_samples = 0
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch, skipped = self._drop_overlong_generation_prompts(test_batch, context="validation")
            skipped_validation_samples += skipped
            if test_batch is None:
                continue
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            # self.checkpoint_manager.sleep_replicas()
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            sample_outputs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids])
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            test_batch = require_reward_fn_key(
                test_batch,
                reward_fn_key=getattr(self.val_reward_fn, "reward_fn_key", self.config.data.get("reward_fn_key", "data_source")),
            )
            scores = self.val_reward_fn.verify(test_batch)
            sample_scores.extend(scores)
            reward_extra_infos_dict["acc"].extend(scores)

            input_ids = test_batch.batch["prompts"]
            sample_inputs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids])
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores))
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        if not data_source_lst:
            return {"val/skipped_overlong_generation_prompts": skipped_validation_samples}
        data_sources = np.concatenate(data_source_lst, axis=0)
        metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        if skipped_validation_samples > 0:
            metrics["val/skipped_overlong_generation_prompts"] = skipped_validation_samples
        return metrics

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_train_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
        )

        if self.use_rm:
            reward_local_path = os.path.join(local_global_step_folder, "reward")
            reward_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "reward")
            )
            self.rm_wg.save_checkpoint(
                reward_local_path,
                reward_remote_path,
                self.global_steps,
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        import dill

        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        reward_path = os.path.join(global_step_folder, "reward")
        # load actor
        self.actor_train_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=False
        )
        # load rm
        if self.use_rm:
            self.rm_wg.load_checkpoint(reward_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        self.train_dataloader = torch.load(dataloader_local_path, weights_only=False)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to
        construct the PPO dataflow. The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):  # self.config.trainer.val_before_train = True
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.global_steps >= self.total_training_steps:
            pprint(
                f"Loaded checkpoint at global step {self.global_steps}, "
                f"which is >= total_training_steps {self.total_training_steps}; skipping training loop."
            )
            return

        # we start from step 1
        self.global_steps += 1
        
        # Support for RL logging board
        rl_logging_board_dir = os.path.join(self.config.trainer.main_path, "rollout_samples", f"{self.config.trainer.project_name}-{self.config.trainer.experiment_name}-{datetime.now(shanghai_tz).strftime('%y%m%d%H')}")
        os.makedirs(rl_logging_board_dir, exist_ok=True)
        rl_logging_board_jsonl = open(os.path.join(rl_logging_board_dir, "data.jsonl"), "w")

        # train
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                rl_logging_board_row = {}
                reward_fn_key = getattr(self.reward_fn, "reward_fn_key", self.config.data.get("reward_fn_key", "data_source"))

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                with simple_timer("step", timing_raw):
                    # generate a batch
                    with simple_timer("gen", timing_raw):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    prompt_group_count = len(batch.batch) // self.config.actor_rollout_ref.rollout.n
                    if prompt_group_count * self.config.actor_rollout_ref.rollout.n != len(batch.batch):
                        raise ValueError(
                            "Repeated rollout batch size must be divisible by rollout.n before DPO group stamping."
                        )
                    batch = batch.union(
                        DataProto.from_dict(
                            tensors={
                                "prompt_group_id": torch.arange(prompt_group_count, dtype=torch.long).repeat_interleave(
                                    self.config.actor_rollout_ref.rollout.n
                                ),
                                "sample_rank_in_group": torch.arange(
                                    self.config.actor_rollout_ref.rollout.n, dtype=torch.long
                                ).repeat(prompt_group_count),
                            }
                        )
                    )
                    batch = batch.union(gen_batch_output)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        if (
                            self.config.reward_model.model.get("loss_type", "ipvrm") == "dpo"
                            and not self.use_prefix_grouper
                        ):
                            raise ValueError(
                                "Standard DPO requires prompt groups to stay on the same DP rank. "
                                "Enable actor_rollout_ref.actor.use_prefix_grouper=true when trainer.balance_batch=true."
                            )
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # verify
                    with simple_timer("verify", timing_raw):
                        batch = require_reward_fn_key(batch, reward_fn_key=reward_fn_key)
                        scores = self.reward_fn.verify(batch)
                        metrics["acc"] = statistics.mean(scores)
                        acc = build_prompt_group_acc_tensor(
                            scores=scores,
                            prompt_group_id=batch.batch["prompt_group_id"],
                            sample_rank_in_group=batch.batch["sample_rank_in_group"],
                        )
                        batch = batch.union(DataProto.from_dict(tensors={"acc": acc}))
                    
                    # filter the batch. 1/oversample_factor samples will be kept.
                    # If there is a filter, prompts passing it will be prioritized.
                    batch, info = self.filter_and_downsample(scores, batch)

                    batch.meta_info["n"] = self.config.actor_rollout_ref.rollout.n
                    n_samples = self.config.actor_rollout_ref.rollout.n
                    if info is not None:
                        metrics.update(info)

                    loss_type = self.config.reward_model.model.get("loss_type", "ipvrm")
                    if loss_type == "dpo":
                        dpo_pair_tensors, dpo_pair_metrics = build_prompt_group_chosen_rejected_tensors(
                            acc=batch.batch["acc"],
                            prompt_group_id=batch.batch["prompt_group_id"],
                            sample_rank_in_group=batch.batch["sample_rank_in_group"],
                            n_samples=n_samples,
                        )
                        batch = batch.union(DataProto.from_dict(tensors=dpo_pair_tensors))
                        metrics.update(dpo_pair_metrics)
                    else:
                        # Build prompt-group tensors for ipvrm/implicitprm after filtering.
                        reward_tensors, reward_tensor_metrics = build_prompt_group_weight_margin_tensors(
                            acc=batch.batch["acc"],
                            prompt_group_id=batch.batch["prompt_group_id"],
                            sample_rank_in_group=batch.batch["sample_rank_in_group"],
                            n_samples=n_samples,
                            base_margin=self.config.reward_model.model.margin,
                            use_dlw=self.config.reward_model.model.get("use_dlw", False),
                            use_adb=self.config.reward_model.model.get("use_adb", False),
                        )
                        batch = batch.union(DataProto.from_dict(tensors=reward_tensors))
                        metrics.update(reward_tensor_metrics)

                    # recompute old_log_probs
                    with simple_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_train_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = compute_response_mask(batch)
                        batch = batch.union(DataProto.from_dict(tensors={"response_mask": response_masks}))
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with simple_timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with simple_timer("adv", timing_raw):
                        if self.use_rm:
                            update_style = self.config.reward_model.model.get("update", "none")
                            if update_style == "none":  # only run forward
                                reward_output = self.rm_wg.compute_rm_score(batch)
                            elif update_style == "after":  # update and directly return the reward
                                reward_output = self.rm_wg.update_rm(batch)
                            elif update_style == "before":  # update reward model, and then run forward
                                reward_output = self.rm_wg.update_rm(batch)
                                if "metrics" in reward_output.meta_info.keys():
                                    reward_output_metrics = reduce_metrics(reward_output.meta_info["metrics"])
                                    metrics.update(reward_output_metrics)
                                reward_output = self.rm_wg.compute_rm_score(batch)
                            elif (
                                update_style == "reverse"
                            ):  # run forward to calculate statistics, then update reward model
                                forward_reward_output = self.rm_wg.compute_rm_score(batch)
                                if "metrics" in forward_reward_output.meta_info.keys():
                                    reward_output_metrics = reduce_metrics(forward_reward_output.meta_info["metrics"])
                                    metrics.update(reward_output_metrics)
                                reward_output = self.rm_wg.update_rm(batch)
                            else:
                                raise NotImplementedError
                            batch = batch.union(reward_output)
                            if "metrics" in reward_output.meta_info.keys():
                                reward_output_metrics = reduce_metrics(reward_output.meta_info["metrics"])
                                metrics.update(reward_output_metrics)

                            # get the mask for distribution-level adv
                            batch = batch.union(build_candidate_token_mask_from_batch(batch, config=self.config))
                            raw_candidate_td_advantages = batch.batch["candidate_td_advantages"]
                            candidate_token_mask = batch.batch["candidate_token_mask"]

                            # compute `advantages` and the covariance between the `logps` and `advantages`
                            metrics.update({"stability/raw_lop_adv_cov": compute_logprob_advantage_correlation(
                                logp=batch.batch["old_log_prob_topk_values"],
                                adv=raw_candidate_td_advantages, 
                                candidate_token_mask=candidate_token_mask)})

                        # compute advantages, executed on the driver process
                        batch, advantage_metrics = compute_advantage(batch, config=self.config)
                        metrics.update(advantage_metrics)
                        
                    # update actor(reward model warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with simple_timer("update_actor", timing_raw):
                            actor_output = self.actor_train_wg.update_actor(batch)
                            self._sync_dist_actor_to_vllm_worker(self.global_steps)
                            self.checkpoint_manager.update_weights(self.global_steps)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    # validate
                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with simple_timer("save_checkpoint", timing_raw):
                            # self._save_checkpoint()
                            self.checkpoint_manager.sleep_replicas()
                            try:
                                self._save_checkpoint()
                            finally:
                                self.checkpoint_manager.update_weights(self.global_steps)
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with simple_timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)
                    
                    # Support for Distribution-Level RL Logging Board
                    max_response_length = batch.batch["responses"].shape[1]
                    max_prompt_length = batch.batch["prompts"].shape[1]
                    step_rewards = batch.batch["step_rewards"]
                    candidate_td_advantages = batch.batch["candidate_td_advantages"]
                    normalized_candidate_td_advantages = batch.batch["normalized_candidate_td_advantages"]

                    ref_log_prob_batch = (
                        batch.batch["ref_log_prob"]
                        if "ref_log_prob" in batch.batch.keys()
                        else batch.batch["old_log_prob"]
                    )
                    ref_log_prob_topk_values_batch = (
                        batch.batch["ref_log_prob_topk_values"]
                        if "ref_log_prob_topk_values" in batch.batch.keys()
                        else batch.batch["old_log_prob_topk_values"]
                    )

                    for (
                        sample_idx,
                        response_mask,
                        prompt_mask,
                        prompt,
                        response,
                        old_log_probs,
                        ref_log_prob,
                        step_reward_values,
                        acc,
                        old_log_prob_topk_indices,
                        sample_candidate_td_advantages,
                        sample_normalized_candidate_td_advantages,
                        old_log_prob_topk_values,
                        ref_log_prob_topk_values,
                        rollout_adv,
                        ) in zip(
                        range(TOPK),
                        batch.batch["attention_mask"][:, -max_response_length:], 
                        batch.batch["attention_mask"][:, :max_prompt_length], 
                        batch.batch["prompts"],
                        batch.batch["responses"], 
                        batch.batch["old_log_prob"], 
                        ref_log_prob_batch,
                        step_rewards,
                        batch.batch["acc"], 
                        batch.batch["old_log_prob_topk_indices"], 
                        candidate_td_advantages, 
                        normalized_candidate_td_advantages,
                        batch.batch["old_log_prob_topk_values"], 
                        ref_log_prob_topk_values_batch, 
                        batch.batch["advantages"], 
                        ):
                        if sample_idx >= MAX_SAMPLES_PER_BATCH:
                            break
                        prompt_mask = prompt_mask.bool()
                        response_mask = response_mask.bool()
                        rl_logging_board_row = {
                            "prompt": self.tokenizer.decode(prompt[prompt_mask]),
                            "response": self.tokenizer.decode(response[response_mask]),
                            "response_tokens": [self.tokenizer.decode(token) for token in response[response_mask.bool()]],
                            "logprobs": old_log_probs[response_mask].detach().numpy().tolist(),
                            "ref_logprobs": ref_log_prob[response_mask].detach().numpy().tolist(),
                            "value": step_reward_values[response_mask].detach().numpy().tolist(),
                            "step_rewards": step_reward_values[response_mask].detach().numpy().tolist(),
                            "token_rewards": rollout_adv[response_mask].detach().numpy().tolist(),
                            "reward": acc.detach().numpy().tolist(),
                            "step": int(self.global_steps),
                        }
                        logged_topk = min(
                            int(self.config.candidate_top_k),
                            int(old_log_prob_topk_indices.shape[-1]),
                            int(sample_candidate_td_advantages.shape[-1]),
                            int(sample_normalized_candidate_td_advantages.shape[-1]),
                            int(old_log_prob_topk_values.shape[-1]),
                            int(ref_log_prob_topk_values.shape[-1]),
                        )
                        for topk_idx in range(logged_topk):
                            rl_logging_board_row[f"top{topk_idx}_tokens"] = [
                                self.tokenizer.decode(token)
                                for token in old_log_prob_topk_indices[-max_response_length:, topk_idx][response_mask]
                            ]
                            rl_logging_board_row[f"top{topk_idx}_advs"] = (
                                sample_candidate_td_advantages[response_mask][:, topk_idx].detach().numpy().tolist()
                            )
                            rl_logging_board_row[f"top{topk_idx}_advs_normed"] = (
                                sample_normalized_candidate_td_advantages[response_mask][:, topk_idx]
                                .detach()
                                .numpy()
                                .tolist()
                            )
                            rl_logging_board_row[f"top{topk_idx}_candidate_td_advantages"] = (
                                sample_candidate_td_advantages[response_mask][:, topk_idx].detach().numpy().tolist()
                            )
                            rl_logging_board_row[f"top{topk_idx}_policy_logp"] = (
                                old_log_prob_topk_values[response_mask][:, topk_idx].detach().numpy().tolist()
                            )
                            rl_logging_board_row[f"top{topk_idx}_ref_logp"] = (
                                ref_log_prob_topk_values[response_mask][:, topk_idx].detach().numpy().tolist()
                            )
                        rl_logging_board_jsonl.write(json.dumps(rl_logging_board_row, ensure_ascii=False) + "\n")
                        

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    if (
                        self.config.trainer.save_freq > 0
                        and (self.global_steps - 1) % self.config.trainer.save_freq != 0
                    ):
                        with simple_timer("save_checkpoint", timing_raw):
                            self.checkpoint_manager.sleep_replicas()
                            self._save_checkpoint()
                    return
                
    def filter_and_downsample_backup(self, scores, batch: DataProto):
        """
        downsample the batch according to oversample_factor
        samples passing the filters will be prioritized
        """
        n_samples = int(self.config.actor_rollout_ref.rollout.n)
        reward_matrix = torch.tensor(scores).reshape(-1, n_samples)

        filter_mask = torch.ones((reward_matrix.shape[0]), dtype=torch.bool)

        if self.config.data.filter_accuracy:
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            filter_mask[
                (acc_tensor > self.config.data.accuracy_upper_bound)
                | (acc_tensor < self.config.data.accuracy_lower_bound)
            ] = False

        if self.config.data.filter_truncate:
            length_matrix = (
                batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1] :]
                .sum(dim=-1)
                .reshape(-1, n_samples)
            )
            length_tensor = torch.max(length_matrix, dim=-1)[0]
            filter_mask[length_tensor >= self.config.data.max_response_length - 1] = False

        reorder_index = torch.argsort(filter_mask, descending=True)
        reorder_index = (reorder_index.unsqueeze(-1) * n_samples + torch.arange(0, n_samples).unsqueeze(0)).view(-1)
        batch.reorder(
            reorder_index[: int(len(batch) // self.config.data.oversample_factor)]
        )  # this operation is inplace

        return batch, None
    

    def filter_and_downsample(self, scores, batch: DataProto):
        """
        downsample the batch according to oversample_factor
        samples passing the filters will be prioritized
        also returns statistics: how many samples were too high, too low, or acceptable in accuracy
        """
        n_samples = int(self.config.actor_rollout_ref.rollout.n)
        if "acc" in batch.batch.keys():
            rewards = batch.batch["acc"].float().view(-1)
        else:
            rewards = torch.as_tensor(scores, dtype=torch.float32).view(-1)
        prompt_group_id = batch.batch["prompt_group_id"].view(-1)
        sample_rank_in_group = batch.batch["sample_rank_in_group"].view(-1)
        if not (rewards.shape == prompt_group_id.shape == sample_rank_in_group.shape):
            raise ValueError(
                "scores, prompt_group_id, and sample_rank_in_group must have the same flattened shape before filtering. "
                f"Got {tuple(rewards.shape)}, {tuple(prompt_group_id.shape)}, and {tuple(sample_rank_in_group.shape)}."
            )

        prompt_ids = torch.unique(prompt_group_id, sorted=True)
        expected_ranks = torch.arange(n_samples, dtype=sample_rank_in_group.dtype, device=sample_rank_in_group.device)
        group_indices = []
        group_acc = []
        group_max_response_length = []
        for prompt_id in prompt_ids:
            indices = torch.nonzero(prompt_group_id == prompt_id, as_tuple=False).flatten()
            if indices.numel() != n_samples:
                raise ValueError(
                    f"Prompt group {prompt_id.item()} has {indices.numel()} samples, expected rollout.n={n_samples}."
                )
            indices = indices[torch.argsort(sample_rank_in_group[indices])]
            ranks = sample_rank_in_group[indices]
            if not torch.equal(ranks, expected_ranks):
                raise ValueError(
                    f"Prompt group {prompt_id.item()} has sample ranks {ranks.tolist()}, expected {expected_ranks.tolist()}."
                )
            group_indices.append(indices.detach().cpu())
            group_acc.append(rewards[indices].mean())
            group_max_response_length.append(
                batch.batch["attention_mask"][indices, -batch.batch["responses"].shape[-1]:].sum(dim=-1).max()
            )

        group_acc = torch.stack(group_acc) if group_acc else torch.empty(0, dtype=torch.float32)
        group_max_response_length = (
            torch.stack(group_max_response_length) if group_max_response_length else torch.empty(0, dtype=torch.long)
        )
        filter_mask = torch.ones((len(group_indices),), dtype=torch.bool, device=group_acc.device)

        # For stats
        n_high, n_low, n_ok = 0, 0, 0

        if self.config.data.filter_accuracy:
            high_mask = group_acc > self.config.data.accuracy_upper_bound
            low_mask = group_acc < self.config.data.accuracy_lower_bound
            ok_mask = ~(high_mask | low_mask)

            filter_mask[high_mask | low_mask] = False

            # Count how many prompts are filtered by each rule.
            n_high = high_mask.sum().item()
            n_low = low_mask.sum().item()
            n_ok = ok_mask.sum().item()
        else:
            n_ok = len(group_indices)

        if self.config.data.filter_truncate:
            trunc_mask = group_max_response_length >= self.config.data.max_response_length - 1
            filter_mask[trunc_mask] = False

        passing_group_positions = torch.nonzero(filter_mask, as_tuple=False).flatten().detach().cpu().tolist()
        failing_group_positions = torch.nonzero(~filter_mask, as_tuple=False).flatten().detach().cpu().tolist()
        ordered_group_positions = passing_group_positions + failing_group_positions
        keep_group_count = int((len(batch) // float(self.config.data.oversample_factor)) // n_samples)
        keep_group_count = min(keep_group_count, len(ordered_group_positions))
        selected_indices = [group_indices[group_pos] for group_pos in ordered_group_positions[:keep_group_count]]
        reorder_index = torch.cat(selected_indices, dim=0) if selected_indices else torch.empty(0, dtype=torch.long)
        batch.reorder(reorder_index)  # inplace operation

        return batch, {"accuracy_too_high": n_high, "accuracy_too_low": n_low, "accuracy_ok": n_ok}

import itertools

import torch
import torch.distributed as dist
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.ulysses import gather_outpus_and_unpad
from .reward_model_core_algos import (
    compute_implicitprm_loss,
    compute_ipvrm_loss,
)
from .reward_signal_utils import (
    build_candidate_td_scores,
    build_step_reward_scores,
)
from .utils import ulysses_pad_and_slice_inputs, chunked_logprobs_from_logits

VALID_REWARD_MODEL_LOSS_TYPES = {"ipvrm", "implicitprm", "dpo"}

__all__ = [
    "DataParallelDistRLRewardModel",
]


# MODIFIED FROM the data-parallel reward model implementation.
class DataParallelDistRLRewardModel:
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        print(f"Reward model use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.loss_type = self.config.model.get("loss_type", "ipvrm")
        if self.loss_type not in VALID_REWARD_MODEL_LOSS_TYPES:
            raise NotImplementedError(f"Unsupported reward-model loss_type: {self.loss_type}")
        self.margin = self.config.model.margin
        self.use_dlw = self.config.model.use_dlw
        self.use_adb = self.config.model.use_adb

    def _validate_prompt_group_layout(self, batch, group_size: int) -> None:
        if "prompt_group_id" not in batch.keys() or "sample_rank_in_group" not in batch.keys():
            raise KeyError(
                "DPO requires tensor fields prompt_group_id and sample_rank_in_group to survive all trainer reordering."
            )

        prompt_group_id = batch["prompt_group_id"].view(-1)
        sample_rank_in_group = batch["sample_rank_in_group"].view(-1)
        if prompt_group_id.numel() % group_size != 0:
            raise ValueError(
                f"Batch size {prompt_group_id.numel()} must be divisible by group_size={group_size} for DPO."
            )

        expected_ranks = torch.arange(
            group_size,
            device=sample_rank_in_group.device,
            dtype=sample_rank_in_group.dtype,
        )
        for start in range(0, prompt_group_id.numel(), group_size):
            stop = start + group_size
            group_ids = prompt_group_id[start:stop]
            if not torch.all(group_ids == group_ids[0]):
                raise ValueError(
                    "DPO prompt groups are no longer contiguous in the local batch. "
                    "This usually means balance_batch or batching split a prompt group before RM update."
                )

            sorted_ranks = torch.sort(sample_rank_in_group[start:stop]).values
            if not torch.equal(sorted_ranks, expected_ranks):
                raise ValueError(
                    "DPO sample_rank_in_group is incomplete inside a local prompt group. "
                    f"Expected ranks {expected_ranks.tolist()}, got {sorted_ranks.tolist()}."
                )

    def _split_grouped_batch(self, batch, split_size: int, group_size: int, split_name: str):
        if group_size == 1:
            return batch.split(split_size)

        if split_size % group_size != 0:
            raise ValueError(
                f"{split_name}={split_size} must be divisible by rollout.n={group_size} for standard DPO."
            )
        if len(batch) % group_size != 0:
            raise ValueError(
                f"Batch size {len(batch)} must be divisible by rollout.n={group_size} for standard DPO."
            )

        self._validate_prompt_group_layout(batch, group_size)
        return [batch[start : start + split_size] for start in range(0, len(batch), split_size)]

    def _get_data_parallel_world_size(self) -> int:
        if not dist.is_initialized():
            return 1
        return max(dist.get_world_size() // max(self.ulysses_sequence_parallel_size, 1), 1)

    def _reduce_global_pair_count(self, local_pair_count: int) -> int:
        pair_count = torch.tensor(float(local_pair_count), device=get_device_name())
        if dist.is_initialized():
            dist.all_reduce(pair_count, op=dist.ReduceOp.SUM)
            if self.ulysses_sequence_parallel_size > 1:
                pair_count /= float(self.ulysses_sequence_parallel_size)
        return int(pair_count.item())

    def _summarize_dpo_prompt_stats(self, prompt_group_id, sample_rank_in_group, chosen_mask, rejected_mask):
        prompt_group_id = prompt_group_id.view(-1)
        sample_rank_in_group = sample_rank_in_group.view(-1)
        chosen_mask = chosen_mask.view(-1).bool()
        rejected_mask = rejected_mask.view(-1).bool()

        valid_prompt_count = 0
        skipped_prompt_count = 0
        unique_prompt_ids = torch.unique(prompt_group_id, sorted=True)
        for prompt_id in unique_prompt_ids:
            group_indices = torch.nonzero(prompt_group_id == prompt_id, as_tuple=False).flatten()
            group_indices = group_indices[torch.argsort(sample_rank_in_group[group_indices])]
            group_pair_count = min(
                int(chosen_mask[group_indices].sum().item()),
                int(rejected_mask[group_indices].sum().item()),
            )
            if group_pair_count > 0:
                valid_prompt_count += 1
            else:
                skipped_prompt_count += 1
        return valid_prompt_count, skipped_prompt_count, int(unique_prompt_ids.numel())

    def _compute_dporm_loss_from_chosen_rejected(
        self,
        token_level_scores,
        response_mask,
        chosen,
        rejected,
        prompt_group_id,
        sample_rank_in_group,
        beta,
    ):
        zero_loss = token_level_scores.sum() * 0.0
        chosen_mask = chosen.view(-1).bool()
        rejected_mask = rejected.view(-1).bool()

        chosen_count = int(chosen_mask.sum().item())
        rejected_count = int(rejected_mask.sum().item())
        if chosen_count != rejected_count:
            raise ValueError(
                "DPO requires chosen and rejected counts to match in each micro batch. "
                f"Got chosen={chosen_count}, rejected={rejected_count}."
            )

        valid_prompt_count, skipped_prompt_count, total_prompt_count = self._summarize_dpo_prompt_stats(
            prompt_group_id=prompt_group_id,
            sample_rank_in_group=sample_rank_in_group,
            chosen_mask=chosen_mask,
            rejected_mask=rejected_mask,
        )

        if chosen_count == 0:
            return zero_loss, {
                "pair_count": 0,
                "valid_prompt_count": valid_prompt_count,
                "skipped_prompt_count": skipped_prompt_count,
                "total_prompt_count": total_prompt_count,
                "pair_loss_mean": 0.0,
            }

        sequence_scores = (token_level_scores * response_mask).sum(dim=1)
        chosen_scores = sequence_scores[chosen_mask]
        rejected_scores = sequence_scores[rejected_mask]
        pair_logits = beta * (chosen_scores - rejected_scores)
        pair_loss = -torch.nn.functional.logsigmoid(pair_logits)
        loss_sum = pair_loss.sum()
        return loss_sum, {
            "pair_count": chosen_count,
            "valid_prompt_count": valid_prompt_count,
            "skipped_prompt_count": skipped_prompt_count,
            "total_prompt_count": total_prompt_count,
            "pair_loss_mean": pair_loss.detach().mean().item(),
        }

    def _forward_micro_batch(self, micro_batch):
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        batch_size, max_seq_length = input_ids.shape
        batch_size, max_response_length = micro_batch['responses'].shape
        max_prompt_length = max_seq_length - max_response_length
        max_positions = attention_mask[:, max_prompt_length:].sum(-1)
        old_log_prob_topk_indices = micro_batch['old_log_prob_topk_indices']
        
        if self.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            output = self.reward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            rm_output_logits = output.logits.squeeze(0)
            rm_log_labels = verl_F.logprobs_from_logits(logits=rm_output_logits, labels=input_ids_rmpad_rolled)

            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            rm_log_labels = pad_input(
                hidden_states=rm_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=max_seq_length
            ).squeeze(-1)[:, -max_response_length - 1 : -1]

            # SUPPORT Distribution-Level Adv
            old_log_prob_topk_indices_rmpad, *_ = unpad_input(old_log_prob_topk_indices.unsqueeze(-1), attention_mask)
            old_log_prob_topk_indices_rmpad = old_log_prob_topk_indices_rmpad.permute(2, 0, 1)
            old_log_prob_topk_indices_rmpad_rolled = torch.roll(old_log_prob_topk_indices_rmpad, shifts=-1, dims=1)
            if self.ulysses_sequence_parallel_size > 1:
                old_log_prob_topk_indices_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    old_log_prob_topk_indices_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                )
                rm_log_prob_topk_values = chunked_logprobs_from_logits(
                    rm_output_logits, old_log_prob_topk_indices_rmpad_rolled[0]
                )
                rm_log_prob_topk_values = gather_outpus_and_unpad(
                    rm_log_prob_topk_values, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
            else:
                old_log_prob_topk_indices_rmpad_rolled = old_log_prob_topk_indices_rmpad_rolled.squeeze(0)
                rm_log_prob_topk_values = chunked_logprobs_from_logits(
                    rm_output_logits, old_log_prob_topk_indices_rmpad_rolled
                )

            if rm_log_prob_topk_values.dim() < 2:
                rm_log_prob_topk_values = rm_log_prob_topk_values.unsqueeze(-1)

            rm_log_prob_topk_values = pad_input(
                hidden_states=rm_log_prob_topk_values,
                indices=indices,
                batch=batch_size,
                seqlen=max_seq_length,
            )[:, -max_response_length - 1:-1]

        else:
            output = self.reward_module(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            rm_output_logits = output.logits
            rm_log_prob = torch.nn.functional.log_softmax(rm_output_logits[:, :-1, :], dim=-1)
            rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

            # SUPPORT Distribution-Level Adv
            rm_log_prob_topk_values = chunked_logprobs_from_logits(
                logits=rm_output_logits,
                labels=torch.roll(old_log_prob_topk_indices, shifts=-1, dims=1)
            )[:, -max_response_length - 1:-1]

            if rm_log_prob_topk_values.dim() < 3:
                rm_log_prob_topk_values = rm_log_prob_topk_values.unsqueeze(-1)

        if self.ref_module is not None:
            # do not have to pad again
            with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output = self.ref_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                    )

                    ref_output_logits = ref_output.logits.squeeze(0)
                    ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits, labels=input_ids_rmpad_rolled)
                    ref_log_labels = gather_outpus_and_unpad(
                        ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    ref_log_labels = pad_input(
                        hidden_states=ref_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=max_seq_length
                    ).squeeze(-1)[:, -max_response_length - 1 : -1]
                    
                    # ref logp for topk
                    ref_log_prob_topk_values = chunked_logprobs_from_logits(ref_output_logits, old_log_prob_topk_indices_rmpad_rolled[0])
                    ref_log_prob_topk_values = gather_outpus_and_unpad(ref_log_prob_topk_values, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                    if ref_log_prob_topk_values.dim() < 2:
                        ref_log_prob_topk_values = ref_log_prob_topk_values.unsqueeze(-1)

                    ref_log_prob_topk_values = pad_input(hidden_states=ref_log_prob_topk_values,
                                                        indices=indices,
                                                        batch=batch_size,
                                                        seqlen=max_seq_length)[:, -max_response_length - 1: -1]
                else:
                    ref_output = self.ref_module(
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch["attention_mask"],
                        position_ids=micro_batch["position_ids"],
                        use_cache=False,
                    )
                    ref_output_logits = ref_output.logits
                    ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits[:, :-1, :], labels=micro_batch["input_ids"][:, 1:].unsqueeze(-1))
                    ref_log_prob_topk_values = chunked_logprobs_from_logits(logits=ref_output_logits, labels=torch.roll(old_log_prob_topk_indices, shifts=-1, dims=1))[:, -max_response_length - 1:-1]
                    
                    if ref_log_prob_topk_values.dim() < 3:
                        ref_log_prob_topk_values = ref_log_prob_topk_values.unsqueeze(-1)
        
        # REWARD SCORE FOR REWARD MODEL TRAINING 
        elif self.config.ref_model_type == "ref":
            ref_log_labels = micro_batch["ref_log_prob"]
        elif self.config.ref_model_type == "old":
            ref_log_labels = micro_batch["old_log_prob"]
        else:
            raise NotImplementedError
        
        ref_log_labels = ref_log_labels.to(rm_log_labels.dtype)
        rm_training_signal = rm_log_labels[:, -max_response_length:] - ref_log_labels[:, -max_response_length:]
        for i in range(micro_batch["input_ids"].shape[0]):
            rm_training_signal[i, max_positions[i] :] = 0


        # REWARD SCORE FOR REINFORCEMENT LEARNING TRAINING 
        if self.ref_module is not None:
            assert 'ref_log_prob_topk_values' in locals(), \
                "ref_log_prob_topk_values should have been computed above when ref_module is not None"
        elif self.config.ref_model_type == "ref":
            ref_log_labels = micro_batch["ref_log_prob"]
            ref_log_prob_topk_values = micro_batch["ref_log_prob_topk_values"]
        elif self.config.ref_model_type == "old":
            ref_log_labels = micro_batch["old_log_prob"]
            ref_log_prob_topk_values = micro_batch["old_log_prob_topk_values"]
        else:
            raise NotImplementedError
        ref_log_labels = ref_log_labels.to(rm_log_labels.dtype)

        with torch.no_grad():
            # calculate the difference
            beta = self.config.model.get("beta_train", 0.05)
            candidate_td_rewards = beta * (rm_log_prob_topk_values[:, -max_response_length:] - ref_log_prob_topk_values[:, -max_response_length:])
            step_td_rewards = beta * (rm_log_labels[:, -max_response_length:] - ref_log_labels[:, -max_response_length:])

            # trim unnecessary logprobs 
            for i in range(micro_batch["input_ids"].shape[0]):
                candidate_td_rewards[i, max_positions[i] :] = 0
                step_td_rewards[i, max_positions[i] :] = 0

            step_reward_scores = build_step_reward_scores(
                step_td_rewards,
                max_positions,
            )
            candidate_td_scores = build_candidate_td_scores(candidate_td_rewards, max_positions)
        return step_reward_scores, rm_training_signal, candidate_td_scores

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip
            )
        self.reward_optimizer.step()
        return grad_norm

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        if self.ref_module is not None:
            self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids", "acc", 
            "old_log_prob", "old_log_prob_topk_indices", "old_log_prob_topk_values", "response_mask",
            "ref_log_prob", "ref_log_prob_topk_values"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        step_reward_scores_lst = []
        rm_training_signal_lst = []
        candidate_td_advantages_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                step_reward_scores, rm_training_signal, candidate_td_advantages = self._forward_micro_batch(micro_batch)
            step_reward_scores_lst.append(step_reward_scores)
            rm_training_signal_lst.append(rm_training_signal)
            candidate_td_advantages_lst.append(candidate_td_advantages)
        step_reward_scores = torch.concat(step_reward_scores_lst, dim=0)
        rm_training_signal = torch.concat(rm_training_signal_lst, dim=0)
        candidate_td_advantages = torch.concat(candidate_td_advantages_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == step_reward_scores.size(0), f"{len(indices)} vs. {step_reward_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            step_reward_scores = step_reward_scores[revert_indices]
            rm_training_signal = rm_training_signal[revert_indices]
            candidate_td_advantages = candidate_td_advantages[revert_indices]
        return (
            step_reward_scores,
            rm_training_signal.detach(),
            candidate_td_advantages,
            {
                "reward_model/reward": step_reward_scores[data.batch['response_mask'].bool()].mean().item(),
                "reward_model/raw_reward": rm_training_signal[data.batch['response_mask'].bool()].mean().item(),
            },
        )

    def update_rm(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}
        beta = self.config.model.get("beta_train", 0.05)
        select_keys = ["prompts", "input_ids", "responses", "attention_mask", "position_ids", "acc", 
            "old_log_prob", "old_log_prob_topk_indices", "old_log_prob_topk_values", "response_mask",
            "ref_log_prob", "ref_log_prob_topk_values"]
        n_samples = data.meta_info["n"]

        if self.loss_type == "dpo":
            select_keys.extend(["prompt_group_id", "sample_rank_in_group", "chosen", "rejected"])
        else:
            select_keys.extend(["dlw_weight", "margin"])

        batch = data.select(batch_keys=select_keys).batch

        if self.loss_type == "dpo":
            self._validate_prompt_group_layout(batch, n_samples)
            group_size = n_samples
        else:
            group_size = 1

        dataloader = self._split_grouped_batch(
            batch=batch,
            split_size=self.config.mini_batch_size,
            group_size=group_size,
            split_name="reward_model.mini_batch_size",
        )
        step_reward_scores_lst = []
        rm_training_signal_lst = []
        candidate_td_advantages_lst = []

        for mini_batch in dataloader:
            if self.loss_type == "dpo":
                chosen_count = int(mini_batch["chosen"].view(-1).bool().sum().item())
                rejected_count = int(mini_batch["rejected"].view(-1).bool().sum().item())
                if chosen_count != rejected_count:
                    raise ValueError(
                        "DPO requires chosen and rejected counts to match in each mini batch. "
                        f"Got chosen={chosen_count}, rejected={rejected_count}."
                    )
                valid_prompt_count, skipped_prompt_count, total_prompt_count = self._summarize_dpo_prompt_stats(
                    prompt_group_id=mini_batch["prompt_group_id"],
                    sample_rank_in_group=mini_batch["sample_rank_in_group"],
                    chosen_mask=mini_batch["chosen"],
                    rejected_mask=mini_batch["rejected"],
                )
                mini_batch_pair_info = {
                    "pair_count": chosen_count,
                    "valid_prompt_count": valid_prompt_count,
                    "skipped_prompt_count": skipped_prompt_count,
                    "total_prompt_count": total_prompt_count,
                }
                global_pair_count = self._reduce_global_pair_count(mini_batch_pair_info["pair_count"])
                dpo_loss_scale = self._get_data_parallel_world_size() / max(float(global_pair_count), 1.0)
            else:
                global_pair_count = None
                dpo_loss_scale = None

            # split batch into micro_batches
            use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)
            if use_dynamic_bsz:
                max_token_len = self.config.get("ppo_max_token_len_per_gpu", 32768) * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch,
                    max_token_len=max_token_len,
                    force_group_size=group_size,
                )
            else:
                micro_batches = self._split_grouped_batch(
                    batch=mini_batch,
                    split_size=self.config.micro_batch_size_per_gpu,
                    group_size=group_size,
                    split_name="reward_model.micro_batch_size_per_gpu",
                )
                indices = None
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()
            mini_batch_step_reward_scores = []
            mini_batch_rm_training_signal = []
            mini_batch_candidate_td_advantages = []

            for micro_batch_data in micro_batches:
                micro_batch_data = micro_batch_data.to(get_device_name())
                attention_mask = micro_batch_data["attention_mask"]
                acc = micro_batch_data["acc"]
                prompt_ids = micro_batch_data["prompts"]
                prompt_length = prompt_ids.shape[-1]
                response_mask = attention_mask[:, prompt_length:]
                step_reward_scores, rm_training_signal, candidate_td_advantages = self._forward_micro_batch(micro_batch_data)
                mini_batch_step_reward_scores.append(step_reward_scores.detach())
                mini_batch_rm_training_signal.append(rm_training_signal.detach())
                mini_batch_candidate_td_advantages.append(candidate_td_advantages.detach())
                
                if self.loss_type == "implicitprm":
                    rm_loss = compute_implicitprm_loss(
                        rm_training_signal,
                        acc,
                        response_mask=response_mask,
                        beta=beta,
                        loss_weight=micro_batch_data["dlw_weight"],
                        margin=micro_batch_data["margin"] if self.use_adb else None,
                    )
                elif self.loss_type == "dpo":
                    rm_loss, dpo_pair_info = self._compute_dporm_loss_from_chosen_rejected(
                        token_level_scores=rm_training_signal,
                        response_mask=response_mask,
                        chosen=micro_batch_data["chosen"],
                        rejected=micro_batch_data["rejected"],
                        prompt_group_id=micro_batch_data["prompt_group_id"],
                        sample_rank_in_group=micro_batch_data["sample_rank_in_group"],
                        beta=beta,
                    )
                elif self.loss_type == "ipvrm":
                    rm_loss = compute_ipvrm_loss(
                        rm_training_signal, 
                        acc, 
                        response_mask=response_mask, 
                        beta=beta, 
                        dlw_weight=micro_batch_data["dlw_weight"], 
                        margin=micro_batch_data["margin"],
                        )
                else:
                    raise NotImplementedError(f"Unsupported reward-model loss_type: {self.loss_type}")

                logged_loss = rm_loss.detach().item()
                metric_data = {
                    "reward_model/loss": logged_loss,
                }
                if self.loss_type == "dpo":
                    metric_data["reward_model/dpo_loss"] = logged_loss
                    metric_data.update(
                        {
                            "reward_model/dpo_pair_count": float(dpo_pair_info["pair_count"]),
                            "reward_model/dpo_valid_prompts": float(dpo_pair_info["valid_prompt_count"]),
                            "reward_model/dpo_skipped_prompts": float(dpo_pair_info["skipped_prompt_count"]),
                            "reward_model/dpo_pair_loss_mean": float(dpo_pair_info["pair_loss_mean"]),
                            "reward_model/dpo_global_pair_count": float(global_pair_count),
                        }
                    )
                    loss = rm_loss * dpo_loss_scale
                    if global_pair_count > 0:
                        loss.backward()
                elif use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = rm_loss * (len(micro_batch_data) / self.config.ppo_mini_batch_size)
                    loss.backward()
                else:
                    loss = rm_loss / self.gradient_accumulation
                    loss.backward()

                append_to_dict(metrics, metric_data)

            if self.loss_type == "dpo" and global_pair_count == 0:
                grad_norm = torch.zeros((), device=get_device_name())
            else:
                grad_norm = self._optimizer_step()
            data = {"reward_model/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)

            step_reward_scores = torch.cat(mini_batch_step_reward_scores, dim=0)
            rm_training_signal = torch.cat(mini_batch_rm_training_signal, dim=0)
            candidate_td_advantages = torch.cat(mini_batch_candidate_td_advantages, dim=0)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == step_reward_scores.size(0), f"{len(indices)} vs. {step_reward_scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                step_reward_scores = step_reward_scores[revert_indices]
                rm_training_signal = rm_training_signal[revert_indices]
                candidate_td_advantages = candidate_td_advantages[revert_indices]

            step_reward_scores_lst.append(step_reward_scores)
            rm_training_signal_lst.append(rm_training_signal)
            candidate_td_advantages_lst.append(candidate_td_advantages)
        self.reward_optimizer.zero_grad()

        step_reward_scores = torch.cat(step_reward_scores_lst, dim=0)
        rm_training_signal = torch.concat(rm_training_signal_lst, dim=0)
        candidate_td_advantages = torch.concat(candidate_td_advantages_lst, dim=0)

        response_mask = batch["response_mask"]
        metrics.update(
            {
            "reward_model/reward": step_reward_scores[response_mask.bool()].mean().item(),
            "reward_model/raw_reward": rm_training_signal[response_mask.bool()].mean().item(),
            },
        )

        return step_reward_scores, candidate_td_advantages, metrics

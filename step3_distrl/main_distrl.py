from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import ray


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from .dl_ray_trainer import RayDistRLTrainer
from .dl_reward_score import step3_prime_verify_compute_score


@hydra.main(config_path="config", config_name="distrl_config", version_base=None)
def main(config) -> None:
    run_distrl(config)


def run_distrl(config):
    if not ray.is_initialized():
        ray_init_kwargs = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "RAY_DEBUG_POST_MORTEM": os.environ.get("RAY_DEBUG_POST_MORTEM", "0"),
                    "RAY_DEBUG_DISABLE_MEMORY_MONITOR": "1",
                    "NCCL_P2P_DISABLE": "1",
                    "NCCL_IB_DISABLE": "1",
                }
            }
        }
        ray_kwargs = config.get("ray_kwargs", {})
        ray_init_config = ray_kwargs.get("ray_init", {}) if ray_kwargs is not None else {}
        num_cpus = ray_init_config.get("num_cpus") if ray_init_config is not None else None
        if num_cpus is not None:
            ray_init_kwargs["num_cpus"] = num_cpus

        ray.init(**ray_init_kwargs)

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        assert config.critic.strategy in {"fsdp", "fsdp2"}
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.engine_workers import ActorRolloutRefWorker

        from .dl_fsdp_workers import DistRLActorRolloutRefWorker, DistRLRewardModelWorker

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Actor: ray.remote(DistRLActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Actor: global_pool_id,
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(DistRLActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(DistRLRewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError

    reward_manager_kwargs = {"compute_score": step3_prime_verify_compute_score}
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, **reward_manager_kwargs)
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, **reward_manager_kwargs)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    config.reward_model.reuse_ref = config.reward_model.model.ref_path == config.actor_rollout_ref.model.path
    trainer = RayDistRLTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        device_name=config.trainer.device,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
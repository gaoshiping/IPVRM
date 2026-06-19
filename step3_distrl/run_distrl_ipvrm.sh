#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
VERL_ROOT=${VERL_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)}

TMPDIR=${TMPDIR:-/tmp/ipvrm}
export TMPDIR
export TEMP=${TEMP:-${TMPDIR}}
export TMP=${TMP:-${TMPDIR}}
export RAY_TMPDIR=${RAY_TMPDIR:-${TMPDIR}/ray}
export HF_HOME=${HF_HOME:-${TMPDIR}/hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_HOME=${TORCH_HOME:-${TMPDIR}/torch}

# Set SWANLAB_API_KEY outside this script when cloud logging is required.
export SWANLAB_MODE=${SWANLAB_MODE:-disabled}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${PROJECT_DIR}/step3_distrl/swanlab}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
RM_CKPT_PATH=${RM_CKPT_PATH:-${PROJECT_DIR}/step2_rm_training/checkpoints/qwen3_0.6b_ipvrm/checkpoint-epoch-2}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/step2_rm_training/data/dapo_math_processed.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/eval_policy/data/val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/step3_distrl/checkpoints/distrl_ipvrm}

IPVRM_PROJECT_ROOT=${PROJECT_DIR} \
IPVRM_WORKSPACE_ROOT=${VERL_ROOT} \
PYTHONPATH="${PROJECT_DIR}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
TOKENIZERS_PARALLELISM=true \
NCCL_SHM_DISABLE=1 \
NCCL_DEBUG=WARN \
RAY_DEBUG_POST_MORTEM=${RAY_DEBUG_POST_MORTEM:-1} \
RAY_DEBUG_DISABLE_MEMORY_MONITOR=1 \
HYDRA_FULL_ERROR=1 \
python -m step3_distrl.main_distrl \
  "paths.project_root=${PROJECT_DIR}" \
  "paths.workspace_root=${VERL_ROOT}" \
  "data.train_files=[${TRAIN_FILE}]" \
  "data.val_files=[${VAL_FILE}]" \
  "actor_rollout_ref.model.path=${MODEL_PATH}" \
  "reward_model.model.path=${RM_CKPT_PATH}" \
  "reward_model.model.tokenizer_path=${RM_CKPT_PATH}" \
  "reward_model.model.ref_path=${MODEL_PATH}" \
  "trainer.default_local_dir=${OUTPUT_DIR}" \
  "paths.output_dir=${OUTPUT_DIR}" \
  "trainer.experiment_name=${EXPERIMENT_NAME:-distrl_ipvrm}" \
  "trainer.nnodes=${NNODES:-1}" \
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-2}" \
  "actor_rollout_ref.actor.distrl_loss_coef=${DISTRL_LOSS_COEF:-0.1}" \
  "algorithm.reward_gae_coef=${REWARD_GAE_COEF:-0.05}" \
  "algorithm.use_kl_in_reward=false" \
  "reward_model.model.update=${RM_UPDATE:-before}" \
  "candidate_top_k=${CANDIDATE_TOP_K:-5}" \
  "trainer.val_before_train=${VAL_BEFORE_TRAIN:-false}"

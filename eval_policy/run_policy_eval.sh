#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  echo "Usage: $0 --model <policy_model_path> [--output-dir <output_dir>] [--n <num_samples>] [--temperature <temperature>] [--top-p <top_p>] [--top-k <top_k>] [--min-p <min_p>]"
  echo "Example: $0 --model step3_distrl/checkpoints/distrl_ipvrm/global_step_16/actor/huggingface --output-dir eval_policy/outputs/distrl_ipvrm_v1 --n 8 --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0"
  echo
  echo "Legacy usage is still supported: $0 <policy_model_path> <output_dir>"
  echo
  echo "Defaults:"
  echo "  --n=8"
  echo "  --temperature=0.7"
  echo "  --top-p=0.8"
  echo "  --top-k=20"
  echo "  --min-p=0"
  echo "  --output-dir=<same path as --model if omitted>"
  echo
  echo "Environment overrides:"
  echo "  DATA_PATH=${DATA_PATH:-<repo>/eval_policy/data/val.parquet}"
  echo "  TOKENIZER_NAME_OR_PATH=<optional tokenizer path>"
  echo "  TENSOR_PARALLEL_SIZE=1"
  echo "  N_GPUS_PER_NODE=<visible gpu count>"
  echo "  GPU_IDS=0,1,2,3"
  echo "  TOP_K=20 MIN_P=0"
  echo "  MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_TOKENS=2048"
  echo "  SCORE_NUM_WORKERS=<cpu workers> SCORING_TIMEOUT_SECONDS=1.0"
  echo "  AUTO_MERGE_FSDP=1 BASE_MODEL_PATH=Qwen/Qwen3-0.6B FORCE_MERGE=0"
}

require_option_value() {
  local option_name=$1
  local option_value=${2:-}
  if [[ -z "${option_value}" || "${option_value}" == --* ]]; then
    echo "Missing value for ${option_name}" >&2
    show_usage
    exit 1
  fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
WORKSPACE_DIR=$(cd "${PROJECT_DIR}/.." && pwd)
CPU_COUNT=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)

is_hf_model_dir() {
  local model_dir=$1
  [[ -d "${model_dir}" ]] || return 1
  [[ -f "${model_dir}/config.json" ]] || return 1

  compgen -G "${model_dir}/*.safetensors" >/dev/null && return 0
  compgen -G "${model_dir}/pytorch_model*.bin" >/dev/null && return 0
  compgen -G "${model_dir}/model*.bin" >/dev/null && return 0
  [[ -f "${model_dir}/model.safetensors.index.json" ]] && return 0
  [[ -f "${model_dir}/pytorch_model.bin.index.json" ]] && return 0
  return 1
}

find_fsdp_actor_dir() {
  local model_path=${1%/}

  if compgen -G "${model_path}/model_world_size_*_rank_*.pt" >/dev/null; then
    printf '%s\n' "${model_path}"
    return 0
  fi

  if [[ -d "${model_path}/actor" ]] && compgen -G "${model_path}/actor/model_world_size_*_rank_*.pt" >/dev/null; then
    printf '%s\n' "${model_path}/actor"
    return 0
  fi

  return 1
}

infer_fsdp_world_size() {
  local actor_dir=$1
  find "${actor_dir}" -maxdepth 1 -name 'model_world_size_*_rank_*.pt' -printf '%f\n' \
    | sed -E 's/model_world_size_([0-9]+)_rank_[0-9]+\.pt/\1/' \
    | sort -u
}

ensure_fsdp_shards_complete() {
  local actor_dir=$1
  local world_size=$2
  local rank
  local kind
  local file

  for rank in $(seq 0 $((world_size - 1))); do
    for kind in model optim extra_state; do
      file="${actor_dir}/${kind}_world_size_${world_size}_rank_${rank}.pt"
      if [[ ! -f "${file}" ]]; then
        echo "ERROR: missing FSDP shard: ${file}" >&2
        return 1
      fi
    done
  done
}

merge_fsdp_if_needed() {
  local input_model_path=${1%/}
  local base_model_path=${BASE_MODEL_PATH:-Qwen/Qwen3-0.6B}
  local auto_merge=${AUTO_MERGE_FSDP:-1}
  local force_merge=${FORCE_MERGE:-0}
  local actor_dir
  local ckpt_dir
  local merge_dir
  local world_sizes
  local world_size

  if is_hf_model_dir "${input_model_path}"; then
    printf '%s\n' "${input_model_path}"
    return 0
  fi

  if [[ "${auto_merge}" != "1" && "${auto_merge}" != "true" ]]; then
    echo "ERROR: ${input_model_path} is not a HuggingFace model dir and AUTO_MERGE_FSDP=${auto_merge}." >&2
    return 1
  fi

  actor_dir=$(find_fsdp_actor_dir "${input_model_path}") || {
    echo "ERROR: ${input_model_path} is neither a HuggingFace model dir nor a supported FSDP checkpoint dir." >&2
    echo "       Expected config.json plus model weights, or model_world_size_*_rank_*.pt files." >&2
    return 1
  }

  if [[ "$(basename "${actor_dir}")" == "actor" ]]; then
    ckpt_dir=$(dirname "${actor_dir}")
  else
    ckpt_dir="${actor_dir}"
  fi
  merge_dir="${ckpt_dir}/merge"

  if [[ "${force_merge}" != "1" && "${force_merge}" != "true" ]] && is_hf_model_dir "${merge_dir}"; then
    echo "Using existing merged HuggingFace model: ${merge_dir}" >&2
    printf '%s\n' "${merge_dir}"
    return 0
  fi

  if [[ ! -d "${base_model_path}" ]]; then
    echo "ERROR: BASE_MODEL_PATH must point to a local HuggingFace model directory for FSDP merge: ${base_model_path}" >&2
    return 1
  fi

  world_sizes=$(infer_fsdp_world_size "${actor_dir}")
  if [[ -z "${world_sizes}" ]]; then
    echo "ERROR: cannot infer FSDP world_size from ${actor_dir}" >&2
    return 1
  fi
  if [[ "$(printf '%s\n' "${world_sizes}" | wc -l)" -ne 1 ]]; then
    echo "ERROR: multiple FSDP world sizes found in ${actor_dir}:" >&2
    printf '%s\n' "${world_sizes}" >&2
    return 1
  fi
  world_size="${world_sizes}"
  ensure_fsdp_shards_complete "${actor_dir}" "${world_size}"

  mkdir -p "${actor_dir}/huggingface"
  find "${base_model_path}" -maxdepth 1 -type f \
    ! -name '*.bin' \
    ! -name '*.safetensors' \
    ! -name '*.pt' \
    ! -name '*.pth' \
    ! -name 'pytorch_model*' \
    ! -name 'model-*' \
    -exec cp -f {} "${actor_dir}/huggingface/" \;

  cat > "${actor_dir}/fsdp_config.json" <<EOF
{
    "FSDP_version": 1,
    "world_size": ${world_size}
}
EOF

  echo "Merging FSDP checkpoint to HuggingFace format" >&2
  echo "  input model: ${input_model_path}" >&2
  echo "  actor dir:   ${actor_dir}" >&2
  echo "  base model:  ${base_model_path}" >&2
  echo "  world size:  ${world_size}" >&2
  echo "  output dir:  ${merge_dir}" >&2

  PYTHONPATH="${WORKSPACE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${actor_dir}" \
    --target_dir "${merge_dir}" >&2

  if ! is_hf_model_dir "${merge_dir}"; then
    echo "ERROR: merge finished but ${merge_dir} does not look like a HuggingFace model dir." >&2
    return 1
  fi

  printf '%s\n' "${merge_dir}"
}

POLICY_MODEL_PATH=
OUTPUT_DIR=
CLI_NUM_SAMPLES=
CLI_TEMPERATURE=
CLI_TOP_P=
CLI_TOP_K=
CLI_MIN_P=

if [[ $# -eq 2 && "${1:-}" != --* && "${2:-}" != --* ]]; then
  POLICY_MODEL_PATH=$1
  OUTPUT_DIR=$2
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        require_option_value "$1" "${2:-}"
        POLICY_MODEL_PATH=$2
        shift 2
        ;;
      --n)
        require_option_value "$1" "${2:-}"
        CLI_NUM_SAMPLES=$2
        shift 2
        ;;
      --temperature)
        require_option_value "$1" "${2:-}"
        CLI_TEMPERATURE=$2
        shift 2
        ;;
      --top-p)
        require_option_value "$1" "${2:-}"
        CLI_TOP_P=$2
        shift 2
        ;;
      --top-k)
        require_option_value "$1" "${2:-}"
        CLI_TOP_K=$2
        shift 2
        ;;
      --min-p)
        require_option_value "$1" "${2:-}"
        CLI_MIN_P=$2
        shift 2
        ;;
      --output-dir)
        require_option_value "$1" "${2:-}"
        OUTPUT_DIR=$2
        shift 2
        ;;
      -h|--help)
        show_usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        show_usage
        exit 1
        ;;
    esac
  done
fi

if [[ -z "${POLICY_MODEL_PATH}" ]]; then
  echo "--model is required" >&2
  show_usage
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${POLICY_MODEL_PATH}"
fi

mkdir -p "${OUTPUT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
POLICY_MODEL_ORIGINAL_PATH="${POLICY_MODEL_PATH}"
POLICY_MODEL_PATH="$(merge_fsdp_if_needed "${POLICY_MODEL_PATH}")"
if [[ "${POLICY_MODEL_PATH}" != "${POLICY_MODEL_ORIGINAL_PATH}" ]]; then
  echo "Using merged policy model for rollout: ${POLICY_MODEL_PATH}"
fi
DATA_PATH=${DATA_PATH:-${SCRIPT_DIR}/data/val.parquet}
DATASET_SPLIT=${DATASET_SPLIT:-}
TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH:-}
PROMPT_KEY=${PROMPT_KEY:-prompt}
REWARD_MODEL_KEY=${REWARD_MODEL_KEY:-reward_model}
RESPONSE_KEY=${RESPONSE_KEY:-responses}
SCORE_KEY=${SCORE_KEY:-score}

NUM_SAMPLES=${CLI_NUM_SAMPLES:-${NUM_SAMPLES:-8}}
TEMPERATURE=${CLI_TEMPERATURE:-${TEMPERATURE:-0.7}}
TOP_P=${CLI_TOP_P:-${TOP_P:-0.8}}
TOP_K=${CLI_TOP_K:-${TOP_K:-20}}
MIN_P=${CLI_MIN_P:-${MIN_P:-0}}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-2048}
PREPROCESS_NUM_PROC=${PREPROCESS_NUM_PROC:-${CPU_COUNT}}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-}
GPU_IDS=${GPU_IDS:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ENABLE_THINKING=${ENABLE_THINKING:-false}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
DISABLE_INIT_EAGER_FALLBACK=${DISABLE_INIT_EAGER_FALLBACK:-false}
BEGIN_INDEX=${BEGIN_INDEX:-0}
END_INDEX=${END_INDEX:-}

LABEL_THRESHOLD=${LABEL_THRESHOLD:-0.5}
SCORING_TIMEOUT_SECONDS=${SCORING_TIMEOUT_SECONDS:-1.0}
SCORE_NUM_WORKERS=${SCORE_NUM_WORKERS:-${CPU_COUNT}}
SCORE_CHUNKSIZE=${SCORE_CHUNKSIZE:-32}

ROLLOUT_PATH="${OUTPUT_DIR}/rollouts_n${NUM_SAMPLES}.json"
SCORED_PATH="${OUTPUT_DIR}/rollouts_n${NUM_SAMPLES}_scored.json"
METRICS_PATH="${OUTPUT_DIR}/metrics_n${NUM_SAMPLES}.json"
REPORT_PATH="${OUTPUT_DIR}/eval_policy.txt"

rollout_cmd=(
  "${PYTHON_BIN}"
  "${PROJECT_DIR}/step2_rm_training/generate_rollouts_vllm.py"
  --data "${DATA_PATH}"
  --prompt-key "${PROMPT_KEY}"
  --begin "${BEGIN_INDEX}"
  --output-path "${ROLLOUT_PATH}"
  --model-name-or-path "${POLICY_MODEL_PATH}"
  --num-samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --min-p "${MIN_P}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS}"
  --preprocess-num-proc "${PREPROCESS_NUM_PROC}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

if [[ -n "${DATASET_SPLIT}" ]]; then
  rollout_cmd+=(--dataset-split "${DATASET_SPLIT}")
fi
if [[ -n "${END_INDEX}" ]]; then
  rollout_cmd+=(--end "${END_INDEX}")
fi
if [[ -n "${TOKENIZER_NAME_OR_PATH}" ]]; then
  rollout_cmd+=(--tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}")
fi
if [[ -n "${N_GPUS_PER_NODE}" ]]; then
  rollout_cmd+=(--n-gpus-per-node "${N_GPUS_PER_NODE}")
fi
if [[ -n "${GPU_IDS}" ]]; then
  rollout_cmd+=(--gpu-ids "${GPU_IDS}")
fi
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  rollout_cmd+=(--enable-thinking)
else
  rollout_cmd+=(--no-enable-thinking)
fi
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  rollout_cmd+=(--enforce-eager)
else
  rollout_cmd+=(--no-enforce-eager)
fi
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  rollout_cmd+=(--trust-remote-code)
fi
if [[ "${DISABLE_INIT_EAGER_FALLBACK}" == "true" ]]; then
  rollout_cmd+=(--disable-init-eager-fallback)
fi

score_cmd=(
  "${PYTHON_BIN}"
  "${PROJECT_DIR}/step2_rm_training/score_rollouts.py"
  --input-path "${ROLLOUT_PATH}"
  --output-path "${SCORED_PATH}"
  --response-key "${RESPONSE_KEY}"
  --reward-model-key "${REWARD_MODEL_KEY}"
  --score-key "${SCORE_KEY}"
  --label-threshold "${LABEL_THRESHOLD}"
  --timeout-seconds "${SCORING_TIMEOUT_SECONDS}"
  --num-workers "${SCORE_NUM_WORKERS}"
  --chunksize "${SCORE_CHUNKSIZE}"
)

echo "[1/3] Generating ${NUM_SAMPLES} rollouts per prompt"
printf 'Command: '
printf '%q ' "${rollout_cmd[@]}"
printf '\n'
"${rollout_cmd[@]}"

echo "[2/3] Scoring sampled rollouts"
printf 'Command: '
printf '%q ' "${score_cmd[@]}"
printf '\n'
"${score_cmd[@]}"

if [[ -f "${SCORED_PATH}" && -f "${ROLLOUT_PATH}" ]]; then
  rm -f "${ROLLOUT_PATH}"
  echo "Removed intermediate rollout json: ${ROLLOUT_PATH}"
fi

echo "[3/3] Aggregating avg@${NUM_SAMPLES} and pass@${NUM_SAMPLES}"
PYTHONPATH="${WORKSPACE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" - \
  "${SCORED_PATH}" \
  "${METRICS_PATH}" \
  "${REPORT_PATH}" \
  "${SCORE_KEY}" \
  "${NUM_SAMPLES}" \
  "${POLICY_MODEL_PATH}" \
  "${DATA_PATH}" \
  "${TEMPERATURE}" \
  "${TOP_P}" \
  "${TOP_K}" \
  "${MIN_P}" \
  "${POLICY_MODEL_ORIGINAL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

from utils.datasets import load_records


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def init_summary():
    return {
        "num_prompts": 0,
        "correct_prompt_sum": 0.0,
        "pass_prompt_sum": 0,
        "total_correct": 0,
        "total_samples": 0,
    }


def update_summary(summary, correct, sample_count):
    summary["num_prompts"] += 1
    summary["correct_prompt_sum"] += correct / sample_count
    summary["pass_prompt_sum"] += int(correct > 0)
    summary["total_correct"] += correct
    summary["total_samples"] += sample_count


def finalize_summary(summary, expected_k):
    num_prompts = summary["num_prompts"]
    avg_at_k = summary["correct_prompt_sum"] / num_prompts if num_prompts else 0.0
    pass_at_k = summary["pass_prompt_sum"] / num_prompts if num_prompts else 0.0
    return {
        "num_prompts": num_prompts,
        "samples_per_prompt": expected_k,
        f"avg@{expected_k}": avg_at_k,
        f"pass@{expected_k}": pass_at_k,
        "total_correct": summary["total_correct"],
        "total_samples": summary["total_samples"],
    }


input_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
score_key = sys.argv[4]
expected_k = int(sys.argv[5])
model_name_or_path = sys.argv[6]
data_path = sys.argv[7]
temperature = float(sys.argv[8])
top_p = float(sys.argv[9])
top_k = int(sys.argv[10])
min_p = float(sys.argv[11])
original_model_name_or_path = sys.argv[12]

records = load_records(input_path)
if not records:
    raise ValueError(f"No scored records found in {input_path}")

overall_summary = init_summary()
grouped_summaries = {}

for record_index, record in enumerate(records):
    scores = [int(score) for score in ensure_list(record.get(score_key))]
    if not scores:
        raise ValueError(f"Record {record_index} has no scores in field '{score_key}'")
    if len(scores) != expected_k:
        raise ValueError(
            f"Record {record_index} has {len(scores)} scores, expected {expected_k}."
        )

    correct = sum(scores)
    update_summary(overall_summary, correct, len(scores))

    data_source = str(record.get("data_source") or "unknown")
    group_summary = grouped_summaries.setdefault(data_source, init_summary())
    update_summary(group_summary, correct, len(scores))

metrics = finalize_summary(overall_summary, expected_k)
grouped_metrics = {
    data_source: finalize_summary(group_summary, expected_k)
    for data_source, group_summary in sorted(grouped_summaries.items())
}

report_lines = [
    "Evaluation Summary",
    f"model: {model_name_or_path}",
    f"original_model: {original_model_name_or_path}",
    f"data: {data_path}",
    f"samples_per_prompt: {expected_k}",
    f"temperature: {temperature}",
    f"top_p: {top_p}",
    f"top_k: {top_k}",
    f"min_p: {min_p}",
    "",
    "Overall",
    f"  num_prompts: {metrics['num_prompts']}",
    f"  avg@{expected_k}: {metrics[f'avg@{expected_k}']:.6f}",
    f"  pass@{expected_k}: {metrics[f'pass@{expected_k}']:.6f}",
    f"  total_correct: {metrics['total_correct']}",
    f"  total_samples: {metrics['total_samples']}",
    "",
    "By data_source",
]

for data_source, data_source_metrics in grouped_metrics.items():
    report_lines.extend(
        [
            f"- {data_source}",
            f"  num_prompts: {data_source_metrics['num_prompts']}",
            f"  avg@{expected_k}: {data_source_metrics[f'avg@{expected_k}']:.6f}",
            f"  pass@{expected_k}: {data_source_metrics[f'pass@{expected_k}']:.6f}",
            f"  total_correct: {data_source_metrics['total_correct']}",
            f"  total_samples: {data_source_metrics['total_samples']}",
        ]
    )

report_text = "\n".join(report_lines) + "\n"

metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
report_path.write_text(report_text, encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(report_text, end="")
PY

echo "Evaluation complete."
echo "  scored json:     ${SCORED_PATH}"
echo "  metrics json:    ${METRICS_PATH}"
echo "  report txt:      ${REPORT_PATH}"

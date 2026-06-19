# IPVRM

Implementation of **IPVRM** for the paper:

> **Unleashing Implicit Rewards: Prefix-Value Learning for Distribution-Level Optimization**

This repository provides a compact implementation of the reward-modeling and policy-optimization pipeline used in the paper. It demonstrates how to train an Implicit Prefix-Value Reward Model with `Qwen/Qwen3-0.6B` on math reasoning data, evaluate reward models, and use the trained reward model for GRPO or DistRL policy optimization.

## Status

The current release includes:

- optional SFT data preparation and training scripts;
- rollout generation with vLLM;
- verifier scoring and reward-model dataset construction;
- IPVRM and baseline reward-model training with Accelerate/FSDP;
- ProcessBench-style reward-model evaluation;
- GRPO and DistRL policy-optimization scripts for VeRL;
- policy evaluation scripts and example result tables.

## Repository Layout

```text
IPVRM/
|-- accelerate_config/
|-- docs/
|   `-- Unleashing_Implicit_Rewards.pdf
|-- eval_policy/
|   |-- data/
|   `-- run_policy_eval.sh
|-- process_bench/
|   |-- data/
|   `-- process_eval.py
|-- step1_sft_training/
|-- step2_rm_training/
|   |-- data/
|   |   |-- dapo_math_processed.parquet
|   |   `-- qwen3_0.6b_rm_data/
|   `-- train_rm_with_accelerate.py
|-- step3_distrl/
|   |-- config/
|   |-- main_distrl.py
|   |-- run_distrl_ipvrm.sh
|   |-- run_grpo_ipvrm.sh
|   `-- run_grpo_vr.sh
|-- utils/
|-- CITATION.cff
|-- LICENSE
|-- README.md
`-- requirements.txt
```

Generated checkpoints, logs, SwanLab/W&B runs, evaluation outputs, Python caches, and temporary rollout files are ignored by default. The parquet files under `step2_rm_training/data/` are intentionally kept in the release.

## Installation

This project is developed with Python 3.12 and is designed to run alongside a local [VeRL](https://github.com/verl-project/verl) checkout.

```bash
git clone https://github.com/verl-project/verl.git
cd verl
pip install --no-deps -e .

git clone https://github.com/gaoshiping/IPVRM.git
cd IPVRM
pip install -r requirements.txt
```

Install PyTorch, vLLM, FlashAttention, and CUDA/NPU runtime packages separately so their versions match your hardware. FlashAttention is required by the default Step 3 configuration; if your environment does not support it, change `actor_rollout_ref.model.override_config.attn_implementation` in `step3_distrl/config/distrl_config.yaml`.

## Data Included in This Release

The repository includes the following parquet files so the reward-model examples can be run without first regenerating all intermediate data:

| File | Purpose |
|---|---|
| `step2_rm_training/data/dapo_math_processed.parquet` | processed math prompts from `open-r1/DAPO-Math-17k-Processed` |
| `step2_rm_training/data/qwen3_0.6b_rm_data/pointwise_train.parquet` | pointwise IPVRM training split |
| `step2_rm_training/data/qwen3_0.6b_rm_data/pointwise_val.parquet` | pointwise IPVRM validation split |
| `step2_rm_training/data/qwen3_0.6b_rm_data/pairwise_train.parquet` | pairwise baseline training split |
| `step2_rm_training/data/qwen3_0.6b_rm_data/pairwise_val.parquet` | pairwise baseline validation split |

These files can be regenerated with the commands in the next section. Generated rollout files such as `dapo_rollouts16.parquet` and `dapo_rollouts16_scored.parquet` are not kept by default.

## Data Preparation

If you want to rebuild the included data from source, run the following steps.

Prepare RL prompts:

```bash
python step2_rm_training/preprocess_dapo17k.py \
  --split train \
  --output-file step2_rm_training/data/dapo_math_processed.parquet
```

Generate multiple responses per prompt:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python step2_rm_training/generate_rollouts_vllm.py \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --data step2_rm_training/data/dapo_math_processed.parquet \
  --output-path step2_rm_training/data/dapo_rollouts16.parquet \
  --n-gpus-per-node 8 \
  --tensor-parallel-size 1 \
  --num-samples 16 \
  --temperature 1 \
  --max-response-tokens 3072
```

Score rollouts with the math verifier:

```bash
python step2_rm_training/score_rollouts.py \
  --input-path step2_rm_training/data/dapo_rollouts16.parquet \
  --output-path step2_rm_training/data/dapo_rollouts16_scored.parquet
```

Build reward-model datasets:

```bash
python step2_rm_training/build_preference_datasets.py \
  --input-path step2_rm_training/data/dapo_rollouts16_scored.parquet \
  --output-dir step2_rm_training/data/qwen3_0.6b_rm_data \
  --pair-strategy best_worst \
  --max-pairs-per-prompt 0
```

Expected schemas:

| Stage | Required fields | Notes |
|---|---|---|
| Rollout input | `prompt`, `reward_model` | `reward_model["ground_truth"]` is required for math verifier scoring. |
| Scored rollout | `prompt`, `responses`, `score` | `responses` and `score` must be aligned lists. |
| Pairwise RM data | `chosen`, `rejected` | Each field is a chat conversation with an assistant answer appended. |
| Pointwise RM data | `conversation`, `label` | `label` is `1` for correct and `0` for incorrect. |

## Reward-Model Training

Train IPVRM on the included pointwise split:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file accelerate_config/fsdp.yaml \
  step2_rm_training/train_rm_with_accelerate.py \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --attn-implementation flash_attention_2 \
  --train-data step2_rm_training/data/qwen3_0.6b_rm_data/pointwise_train.parquet \
  --val-data step2_rm_training/data/qwen3_0.6b_rm_data/pointwise_val.parquet \
  --output-dir step2_rm_training/checkpoints/qwen3_0.6b_ipvrm \
  --reward-type logp_ratio \
  --loss-type ipvrm \
  --max-length 2560 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3 \
  --learning-rate 5e-7 \
  --save-strategy epoch \
  --beta 10
```

The same trainer supports baseline reward-model objectives:

| Model | Training data | `reward-type` | `loss-type` |
|---|---|---|---|
| DPO-RM | pairwise | `logp_ratio` | `bt_sum` |
| Implicit PRM | pointwise | `logp_ratio` | `implicit_prm` |
| QRM | pairwise | `value_head` | `bt_mean` |
| IPVRM | pointwise | `logp_ratio` | `ipvrm` |

For `reward-type=logp_ratio`, pass `--ref-model-name-or-path` to use a separate reference model. If omitted, the trainer uses the main model path as the reference.

## Reward-Model Evaluation

`process_bench/process_eval.py` supports reward inference and threshold-based evaluation on JSON files in `process_bench/data/`.

For IPVRM and other log-probability-ratio reward models:

```bash
CUDA_VISIBLE_DEVICES=0,1 python process_bench/process_eval.py \
  --mode inference \
  --model_path step2_rm_training/checkpoints/qwen3_0.6b_ipvrm/checkpoint-epoch-2 \
  --ref_model_path Qwen/Qwen3-0.6B \
  --scoring_mode logp_ratio \
  --dataset_dir process_bench/data \
  --dataset_names gsm8k math olympiadbench omnimath \
  --output_dir step2_rm_training/checkpoints/qwen3_0.6b_ipvrm/checkpoint-epoch-2/results
```

Recompute metrics from saved reward files:

```bash
python process_bench/process_eval.py \
  --mode evaluate \
  --dataset_names gsm8k math olympiadbench omnimath \
  --input_dir step2_rm_training/checkpoints/qwen3_0.6b_ipvrm/checkpoint-epoch-2/results
```

Example F1 scores from the paper setup:

| Model | Best epoch | GSM8K | MATH | OlympiadBench | OmniMath | Average F1 |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3_0.6b_ipvrm` | epoch-2 | 0.4264 | 0.4494 | 0.3819 | 0.4258 | 0.4209 |
| `qwen3_0.6b_dporm` | epoch-2 | 0.4024 | 0.3672 | 0.3078 | 0.2980 | 0.3439 |
| `qwen3_0.6b_qrm` | epoch-1 | 0.3734 | 0.3405 | 0.3153 | 0.2638 | 0.3232 |
| `qwen3_0.6b_implicit_prm` | epoch-3 | 0.3130 | 0.2934 | 0.2300 | 0.2365 | 0.2682 |

## Policy Optimization

Step 3 runs through VeRL. The scripts are environment-variable driven and should be launched from the repository root.

Common variables:

| Variable | Default | Meaning |
|---|---|---|
| `VERL_ROOT` | parent directory of this repository | local VeRL checkout root |
| `MODEL_PATH` | `Qwen/Qwen3-0.6B` | actor base or SFT model |
| `RM_CKPT_PATH` | `step2_rm_training/checkpoints/qwen3_0.6b_ipvrm/checkpoint-epoch-2` | trained IPVRM checkpoint |
| `TRAIN_FILE` | `step2_rm_training/data/dapo_math_processed.parquet` | RL training prompts |
| `VAL_FILE` | `eval_policy/data/val.parquet` | validation prompts |
| `OUTPUT_DIR` | script-specific checkpoint directory | VeRL checkpoint output |
| `SWANLAB_MODE` | `disabled` | logging mode; set `SWANLAB_API_KEY` outside the script for cloud logging |

Run DistRL + IPVRM:

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-0.6B \
RM_CKPT_PATH=step2_rm_training/checkpoints/qwen3_0.6b_ipvrm/checkpoint-epoch-2 \
bash step3_distrl/run_distrl_ipvrm.sh
```

Run GRPO + IPVRM:

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=Qwen/Qwen3-0.6B \
RM_CKPT_PATH=step2_rm_training/checkpoints/qwen3_0.6b_ipvrm/checkpoint-epoch-2 \
bash step3_distrl/run_grpo_ipvrm.sh
```

The two scripts differ mainly in the actor update:

| Setting | GRPO + IPVRM | DistRL + IPVRM |
|---|---:|---:|
| `actor_rollout_ref.actor.distrl_loss_coef` | `0` | `0.1` |
| `candidate_top_k` | `1` | `5` |

The actor loss is:

```python
policy_loss = pg_loss * pg_loss_coef + distrl_loss_coef * topk_pg_loss
```

With `distrl_loss_coef=0`, the run is the GRPO + IPVRM baseline. With `distrl_loss_coef=0.1` and `candidate_top_k=5`, DistRL uses IPVRM-derived one-step TD advantages for high-probability candidate tokens, giving denser distribution-level supervision without extra full rollouts.

## Policy Evaluation

Evaluate a saved policy checkpoint:

```bash
bash eval_policy/run_policy_eval.sh \
  --model step3_distrl/checkpoints/grpo_ipvrm_0.05/global_step_16/actor/huggingface \
  --output-dir eval_policy/outputs/qwen3_grpo_ipvrm

bash eval_policy/run_policy_eval.sh \
  --model step3_distrl/checkpoints/distrl_ipvrm/global_step_16/actor/huggingface \
  --output-dir eval_policy/outputs/qwen3_distrl_ipvrm
```

If `--model` points to an FSDP checkpoint instead of a Hugging Face directory, set `BASE_MODEL_PATH` to a local Hugging Face model directory before running the script.

The script writes:

- `rollouts_n8_scored.json`: sampled responses with verifier scores;
- `metrics_n8.json`: overall `avg@8` and `pass@8`;
- `eval_policy.txt`: readable summary with per-benchmark results.

Example policy evaluation results:

| Method | aimo-aime2024 | aimo-amc | math-500 | minerva-math | olympiad-bench | AVG |
|---|---:|---:|---:|---:|---:|---:|
| GRPO + IPVRM 0.05 | 2.50% | 22.59% | 46.15% | 12.73% | 16.54% | 20.10% |
| DistRL + IPVRM 0.05 | 2.78% | 23.49% | 53.10% | 16.04% | 20.41% | 23.16% |
| Absolute gain | +0.28 pp | +0.90 pp | +6.95 pp | +3.31 pp | +3.87 pp | +3.06 pp |

## Troubleshooting

- If `math_verify` is missing, install `math-verify` before scoring rollouts.
- If FlashAttention is unavailable, change the attention implementation in config or install a build compatible with your PyTorch/CUDA stack.
- If FSDP policy evaluation fails during merge, set `BASE_MODEL_PATH` to a local Hugging Face model directory.
- If VeRL imports fail, set `VERL_ROOT` to the root of your local VeRL checkout and make sure `pip install --no-deps -e .` was run there.
- If SwanLab cloud logging is needed, export `SWANLAB_API_KEY` in your shell. Do not commit API keys to the repository.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{gao2026unleashing,
  title     = {Unleashing Implicit Rewards: Prefix-Value Learning for Distribution-Level Optimization},
  author    = {Gao, Shiping and Chen, Hongzhan and Quan, Xiaojun and Wang, Qifan and Huang, Lifu},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)},
  year      = {2026}
}
```

## Acknowledgements

This implementation builds on VeRL, vLLM, Hugging Face Transformers, Accelerate, PyTorch, and math-verification tooling. We thank the maintainers of these projects.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

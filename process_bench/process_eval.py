import argparse
import gc
import json
import multiprocessing as mp
import os
from math import sqrt
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


###############################
# PART 1: SIGMOID & EVALUATION
###############################

BATCH_SIZE = 2
COEF = 0.001
SCORING_MODES = ("logp", "logp_ratio", "value_head")
DEFAULT_DATASET_NAMES = ("gsm8k", "math", "olympiadbench", "omnimath")
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def find_threshold_sigmoid(data, num_thresholds=10000):
    scores = []
    for d in data:
        scores.append([sigmoid(s) for s in d["reward"]])
    return find_threshold_base(scores, data, num_thresholds)


def find_threshold_rewardsum_minus_sigmoid(data, num_thresholds=10000):
    scores = []
    for d in data:
        tmp = [0.0] + d["reward"]
        scores.append([sigmoid(tmp[i + 1] - tmp[i]) for i in range(len(tmp) - 1)])
    return find_threshold_base(scores, data, num_thresholds)


def find_threshold_mean_sigmoid(data, num_thresholds=10000):
    scores = []
    for d in data:
        scores.append([sigmoid(a / sqrt(COEF)) for a in d["avg_reward"]])
    return find_threshold_base(scores, data, num_thresholds)


def find_threshold_meandiff_sigmoid(data, num_thresholds=10000):
    scores = []
    for d in data:
        s_list = [0.0] + d["reward"]
        l_list = [0]

        for s, a in zip(d["reward"], d["avg_reward"]):
            if abs(a) > 1e-9:
                l_list.append(round(s / a))
            else:
                l_list.append(l_list[-1] + 1)

        step_scores = []
        for i in range(len(d["reward"])):
            step_reward = s_list[i + 1] - s_list[i]
            step_len = l_list[i + 1] - l_list[i]
            if step_len <= 0:
                step_len = 1
            step_mean_diff = step_reward / step_len
            step_scores.append(sigmoid(step_mean_diff / sqrt(COEF)))

        scores.append(step_scores)

    return find_threshold_base(scores, data, num_thresholds)


def find_threshold_base(sigmoid_scores: List[List[float]], data, num_thresholds=10000):
    thresholds = np.linspace(-1, 1, num_thresholds)
    best_threshold = None
    best_f1 = 0.0

    for t in thresholds:
        for d, sigmoid_score in zip(data, sigmoid_scores):
            pred_step = -1
            for i, sc in enumerate(sigmoid_score):
                if sc < t:
                    pred_step = i
                    break
            d["match"] = d["label"] == pred_step

        correct_data = [x for x in data if x["label"] == -1]
        error_data = [x for x in data if x["label"] != -1]
        if len(correct_data) == 0 or len(error_data) == 0:
            continue

        acc_1 = sum(x["match"] for x in correct_data) / len(correct_data)
        acc_2 = sum(x["match"] for x in error_data) / len(error_data)

        if (acc_1 + acc_2) > 0:
            f1_metric = 2.0 * acc_1 * acc_2 / (acc_1 + acc_2)
        else:
            f1_metric = 0.0

        if best_threshold is None or f1_metric > best_f1:
            best_f1 = f1_metric
            best_threshold = t

    return best_threshold, best_f1


###############################
# PART 2: SHARED INFERENCE UTILS
###############################


def load_tokenizer(tokenizer_path: str):
    print("Loading tokenizer from:", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_dataset_names(dataset_names: Optional[List[str]]) -> Optional[List[str]]:
    if not dataset_names:
        return None

    resolved_names: List[str] = []
    for name in dataset_names:
        if name == "all":
            for default_name in DEFAULT_DATASET_NAMES:
                if default_name not in resolved_names:
                    resolved_names.append(default_name)
        elif name not in resolved_names:
            resolved_names.append(name)

    return resolved_names


def resolve_inference_jobs(args) -> List[Dict[str, str]]:
    dataset_names = normalize_dataset_names(args.dataset_names)
    if dataset_names:
        if not args.output_dir:
            raise ValueError("--output_dir is required when --dataset_names is used in inference mode.")

        os.makedirs(args.output_dir, exist_ok=True)
        jobs = []
        for dataset_name in dataset_names:
            jobs.append(
                {
                    "dataset_name": dataset_name,
                    "input_file": os.path.join(args.dataset_dir, f"{dataset_name}.json"),
                    "output_file": os.path.join(args.output_dir, f"{dataset_name}_rewards.json"),
                }
            )
        return jobs

    if not (args.input_file and args.output_file):
        raise ValueError(
            "Please specify either --input_file/--output_file or --dataset_names/--output_dir for inference mode."
        )

    return [
        {
            "dataset_name": os.path.splitext(os.path.basename(args.input_file))[0],
            "input_file": args.input_file,
            "output_file": args.output_file,
        }
    ]


def resolve_evaluate_jobs(args) -> List[Dict[str, str]]:
    dataset_names = normalize_dataset_names(args.dataset_names)
    if dataset_names:
        if not args.input_dir:
            raise ValueError("--input_dir is required when --dataset_names is used in evaluate mode.")

        jobs = []
        for dataset_name in dataset_names:
            jobs.append(
                {
                    "dataset_name": dataset_name,
                    "input_file": os.path.join(args.input_dir, f"{dataset_name}_rewards.json"),
                }
            )
        return jobs

    if not args.input_file:
        raise ValueError("Please specify either --input_file or --dataset_names/--input_dir for evaluate mode.")

    return [
        {
            "dataset_name": os.path.splitext(os.path.basename(args.input_file))[0],
            "input_file": args.input_file,
        }
    ]


def load_processbench_data(input_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for d in data:
        d["query"] = d["problem"]
        d["answer"] = [f"Step {i + 1}: " + step for i, step in enumerate(d["steps"])]

    return data


def build_item_tensors(item, tokenizer, step_index_offset: int):
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": item["query"]},
            {"role": "assistant", "content": "\n\n".join(item["answer"])},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).long()

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    step_last_tokens = []
    for step_num in range(len(item["answer"]) + 1):
        conv = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item["query"]},
                {"role": "assistant", "content": "\n\n".join(item["answer"][:step_num])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        ).strip()
        if step_num != 0 and step_num != len(item["answer"]):
            conv += "\n\n"
        conv_ids = tokenizer.encode(conv, add_special_tokens=False)
        step_last_tokens.append(max(0, len(conv_ids) - step_index_offset))

    labels = input_ids.clone()
    return input_ids, attention_mask, labels, step_last_tokens


def collate_fn(batch_items, tokenizer):
    pad_id = tokenizer.pad_token_id
    max_len = max(x[0].shape[0] for x in batch_items)

    input_ids_list, attn_mask_list, labels_list = [], [], []
    step_positions_list, raw_items_list = [], []

    for inp, attn, lab, step_pos, raw_item in batch_items:
        pad_len = max_len - inp.shape[0]
        padded_inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=inp.dtype)])
        padded_attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
        padded_lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])

        input_ids_list.append(padded_inp.unsqueeze(0))
        attn_mask_list.append(padded_attn.unsqueeze(0))
        labels_list.append(padded_lab.unsqueeze(0))
        step_positions_list.append([int(max(0, min(s, max_len - 1))) for s in step_pos])
        raw_items_list.append(raw_item)

    return {
        "input_ids": torch.cat(input_ids_list, dim=0).long(),
        "attention_mask": torch.cat(attn_mask_list, dim=0).long(),
        "labels": torch.cat(labels_list, dim=0).long(),
        "step_positions": step_positions_list,
        "raw_items": raw_items_list,
    }


def save_results(output_file: str, results):
    if not output_file:
        raise ValueError("Please specify --output_file for inference mode.")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Writing output to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def parse_visible_devices() -> List[str]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_value:
        return [device.strip() for device in raw_value.split(",") if device.strip()]

    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def split_jobs_across_workers(jobs: List[Dict[str, str]], num_workers: int) -> List[List[Dict[str, str]]]:
    buckets: List[List[Dict[str, str]]] = [[] for _ in range(num_workers)]
    for index, job in enumerate(jobs):
        buckets[index % num_workers].append(job)
    return buckets


###############################
# PART 3: LM-HEAD SCORING
###############################


def setup_accelerator_and_lm(model_path: str):
    accelerator = Accelerator()
    print("Loading model from:", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": accelerator.process_index},
    )
    model.eval()
    model = accelerator.prepare(model)
    return accelerator, model


def get_logps(model, inputs):
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["labels"][:, 1:].clone().long()
    shift_labels[shift_labels == -100] = 0

    per_token_logps = []
    for i in range(shift_logits.size(0)):
        log_probs = shift_logits[i].log_softmax(-1)
        per_token_logps_single = torch.gather(
            log_probs,
            dim=1,
            index=shift_labels[i].unsqueeze(1),
        ).squeeze(1)
        per_token_logps.append(per_token_logps_single.unsqueeze(0))

    return torch.cat(per_token_logps, dim=0)


def run_lm_inference(args, input_file: str, output_file: str):
    tokenizer = load_tokenizer(args.tokenizer_path)
    data = load_processbench_data(input_file)
    data_tuples = []

    for d in data:
        inp, attn, lab, steps = build_item_tensors(d, tokenizer, step_index_offset=2)
        data_tuples.append((inp, attn, lab, steps, d))

    accelerator, model = setup_accelerator_and_lm(args.model_path)

    all_main_logps = []
    print(f"Computing main model token logps for scoring_mode={args.scoring_mode}...")
    for i in tqdm(range(0, len(data_tuples), args.batch_size)):
        batch_slice = data_tuples[i : i + args.batch_size]
        batch = collate_fn(batch_slice, tokenizer)
        model_inputs = {
            "input_ids": batch["input_ids"].to(accelerator.device),
            "attention_mask": batch["attention_mask"].to(accelerator.device),
            "labels": batch["labels"].to(accelerator.device),
        }
        with torch.no_grad():
            all_main_logps.append(get_logps(model, model_inputs).cpu())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ref_model = None
    if args.scoring_mode == "logp_ratio":
        accelerator, ref_model = setup_accelerator_and_lm(args.ref_model_path)

    results = []
    logp_idx = 0
    if args.scoring_mode == "logp_ratio":
        print("Computing reference model token logps...")
    else:
        print("Aggregating logp rewards without reference model...")

    for i in tqdm(range(0, len(data_tuples), args.batch_size)):
        batch_slice = data_tuples[i : i + args.batch_size]
        batch = collate_fn(batch_slice, tokenizer)

        if ref_model is not None:
            model_inputs = {
                "input_ids": batch["input_ids"].to(accelerator.device),
                "attention_mask": batch["attention_mask"].to(accelerator.device),
                "labels": batch["labels"].to(accelerator.device),
            }
            with torch.no_grad():
                ref_logps = get_logps(ref_model, model_inputs).cpu()
        else:
            ref_logps = None

        main_logps = all_main_logps[logp_idx]
        logp_idx += 1
        raw_reward = main_logps if ref_logps is None else main_logps - ref_logps

        for idx_in_batch, (_, _, _, step_pos, raw_item) in enumerate(batch_slice):
            seq_len = batch["input_ids"].shape[1]
            first_boundary = max(0, min(step_pos[0], seq_len - 2))

            mask = torch.zeros(seq_len - 1, dtype=torch.float32)
            mask[first_boundary:] = 1.0

            item_raw_reward = raw_reward[idx_in_batch]
            weighted_reward = args.coef * item_raw_reward * mask
            csum = weighted_reward.cumsum(dim=-1)

            gather_indices = torch.tensor(step_pos[1:], dtype=torch.long)
            gather_indices = torch.clamp(gather_indices, 0, seq_len - 2)
            final_values = csum.gather(dim=-1, index=gather_indices)
            final_values_list = final_values.tolist()

            avg_rewards_list = []
            for step_idx, step_end_idx in enumerate(step_pos[1:]):
                resp_len = step_end_idx - first_boundary
                if resp_len > 0:
                    avg_rewards_list.append(final_values_list[step_idx] / resp_len)
                else:
                    avg_rewards_list.append(0.0)

            raw_item["reward"] = final_values_list
            raw_item["avg_reward"] = avg_rewards_list
            results.append(raw_item)

    save_results(output_file, results)


###############################
# PART 4: VALUE-HEAD SCORING
###############################


def infer_hidden_size(config) -> int:
    for attribute in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attribute):
            return int(getattr(config, attribute))
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attribute in ("hidden_size", "n_embd", "d_model"):
            if hasattr(text_config, attribute):
                return int(getattr(text_config, attribute))
    raise ValueError("Unable to infer hidden size for the reward head.")


def load_reward_head_state(model_path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    reward_head_path = os.path.join(model_path, "reward_head.pt")
    if os.path.exists(reward_head_path):
        state_dict = torch.load(reward_head_path, map_location="cpu")
        return {key: value.float() for key, value in state_dict.items()}

    value_head_path = os.path.join(model_path, "value_head.safetensors")
    if os.path.exists(value_head_path):
        state_dict = load_file(value_head_path, device="cpu")
    else:
        legacy_value_head_path = os.path.join(model_path, "value_head.bin")
        if not os.path.exists(legacy_value_head_path):
            raise FileNotFoundError(
                f"No reward head file found under {model_path}. "
                "Expected reward_head.pt, value_head.safetensors, or value_head.bin."
            )
        state_dict = torch.load(legacy_value_head_path, map_location="cpu")

    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("v_head."):
            normalized_key = normalized_key[len("v_head.") :]
        if normalized_key.startswith("summary."):
            normalized_key = normalized_key[len("summary.") :]
        normalized_state_dict[normalized_key] = value.float()
    return normalized_state_dict


class CausalLMWithRewardHead(nn.Module):
    def __init__(self, base_model: nn.Module, reward_head_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.base_model = base_model
        hidden_size = infer_hidden_size(base_model.config)
        has_bias = "bias" in reward_head_state
        self.reward_head = nn.Linear(hidden_size, 1, bias=has_bias)
        self.reward_head.load_state_dict(reward_head_state, strict=True)
        base_dtype = next(base_model.parameters()).dtype
        self.reward_head = self.reward_head.to(dtype=base_dtype)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_hidden_state = outputs.hidden_states[-1]
        target_device = self.reward_head.weight.device
        target_dtype = self.reward_head.weight.dtype
        if last_hidden_state.device != target_device or last_hidden_state.dtype != target_dtype:
            last_hidden_state = last_hidden_state.to(device=target_device, dtype=target_dtype)
        values = self.reward_head(last_hidden_state).squeeze(-1)
        return values


def setup_accelerator_and_value_model(model_path: str):
    accelerator = Accelerator()
    print("Loading value-head model from:", model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": accelerator.process_index},
    )
    reward_head_state = load_reward_head_state(model_path)
    model = CausalLMWithRewardHead(base_model, reward_head_state)
    model.eval()
    model = accelerator.prepare(model)
    return accelerator, model


def get_values(model, inputs):
    with torch.no_grad():
        values = model(**inputs)
    return values


def run_value_head_inference(args, input_file: str, output_file: str):
    tokenizer = load_tokenizer(args.tokenizer_path)
    data = load_processbench_data(input_file)
    data_tuples = []

    for d in data:
        inp, attn, lab, steps = build_item_tensors(d, tokenizer, step_index_offset=1)
        data_tuples.append((inp, attn, lab, steps, d))

    accelerator, model = setup_accelerator_and_value_model(args.model_path)

    results = []
    print("Computing value-head rewards...")
    for i in tqdm(range(0, len(data_tuples), args.batch_size)):
        batch_slice = data_tuples[i : i + args.batch_size]
        batch = collate_fn(batch_slice, tokenizer)

        model_inputs = {
            "input_ids": batch["input_ids"].to(accelerator.device),
            "attention_mask": batch["attention_mask"].to(accelerator.device),
        }
        values = get_values(model, model_inputs)

        for idx_in_batch, (_, _, _, _, raw_item) in enumerate(batch_slice):
            seq_len = batch["input_ids"].shape[1]
            step_pos = batch["step_positions"][idx_in_batch]
            first_boundary = max(0, min(step_pos[0], seq_len - 1))

            mask = torch.zeros(seq_len, dtype=torch.float32, device=accelerator.device)
            mask[first_boundary:] = 1.0

            item_rewards = values[idx_in_batch] * mask
            csum = item_rewards.cumsum(dim=-1)

            gather_indices = torch.tensor(step_pos[1:], device=accelerator.device)
            gather_indices = torch.clamp(gather_indices, 0, seq_len - 1)
            final_values = csum.gather(dim=-1, index=gather_indices)
            final_values_list = (final_values * args.coef).cpu().tolist()

            avg_rewards_list = []
            for step_idx, step_end_idx in enumerate(step_pos[1:]):
                resp_len = step_end_idx - first_boundary
                if resp_len > 0:
                    avg_rewards_list.append(final_values_list[step_idx] / resp_len)
                else:
                    avg_rewards_list.append(0.0)

            raw_item["reward"] = final_values_list
            raw_item["avg_reward"] = avg_rewards_list
            results.append(raw_item)

    save_results(output_file, results)


###############################
# PART 5: MAIN
###############################


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_evaluation_metrics(input_file: str, dataset_name: Optional[str], num_thresholds: int) -> List[str]:
    lines: List[str] = []
    if dataset_name:
        lines.append(f"===== Evaluating {dataset_name} =====")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_th, best_f1 = find_threshold_sigmoid(data, num_thresholds=num_thresholds)
    lines.append(f"Best threshold found (sum): {format_optional_float(best_th)}, F1={best_f1:.4f}")

    best_th, best_f1 = find_threshold_rewardsum_minus_sigmoid(data, num_thresholds=num_thresholds)
    lines.append(f"Best threshold found (sum_diff): {format_optional_float(best_th)}, F1={best_f1:.4f}")

    best_th, best_f1 = find_threshold_mean_sigmoid(data, num_thresholds=num_thresholds)
    lines.append(f"Best threshold found (mean): {format_optional_float(best_th)}, F1={best_f1:.4f}")

    best_th, best_f1 = find_threshold_meandiff_sigmoid(data, num_thresholds=num_thresholds)
    lines.append(f"Best threshold found (mean_diff): {format_optional_float(best_th)}, F1={best_f1:.4f}")
    return lines


def evaluate_file(input_file: str, dataset_name: Optional[str], num_thresholds: int) -> List[str]:
    lines = format_evaluation_metrics(input_file, dataset_name, num_thresholds)
    for line in lines:
        print(line)
    return lines


def resolve_report_dir(args) -> Optional[str]:
    if args.output_dir:
        return args.output_dir
    if args.mode in {"evaluate", "validate"} and args.input_dir:
        return args.input_dir
    if args.mode == "inference" and args.output_file:
        return os.path.dirname(args.output_file) or "."
    if args.mode in {"evaluate", "validate"} and args.input_file:
        return os.path.dirname(args.input_file) or "."
    return None


def write_summary_report(report_dir: Optional[str], lines: List[str]) -> None:
    if not report_dir:
        return
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "result.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Writing summary report to {report_file}")


def print_evaluation_summary(jobs: List[Dict[str, str]], num_thresholds: int, report_dir: Optional[str] = None) -> List[str]:
    summary_lines = ["", "===== Benchmark Results ====="]
    print("\n===== Benchmark Results =====")
    for job in jobs:
        input_file = job.get("output_file", job.get("input_file"))
        if input_file is None:
            raise ValueError("Each evaluation job must provide either output_file or input_file.")
        job_lines = evaluate_file(input_file, job["dataset_name"], num_thresholds)
        summary_lines.extend(job_lines)

    write_summary_report(report_dir, summary_lines[1:])
    return summary_lines[1:]


def run_inference_jobs(args):
    jobs = resolve_inference_jobs(args)
    for job in jobs:
        print(f"===== Running inference for {job['dataset_name']} =====")
        if args.scoring_mode in {"logp", "logp_ratio"}:
            run_lm_inference(args, job["input_file"], job["output_file"])
        else:
            run_value_head_inference(args, job["input_file"], job["output_file"])


def run_single_inference_job(args, job: Dict[str, str]) -> None:
    print(f"===== Running inference for {job['dataset_name']} =====")
    if args.scoring_mode in {"logp", "logp_ratio"}:
        run_lm_inference(args, job["input_file"], job["output_file"])
    else:
        run_value_head_inference(args, job["input_file"], job["output_file"])


def inference_worker(worker_id: int, visible_device: str, jobs: List[Dict[str, str]], args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    print(
        f"[worker {worker_id}] bound to CUDA_VISIBLE_DEVICES={visible_device} "
        f"and received {len(jobs)} job(s)."
    )
    for job in jobs:
        run_single_inference_job(args, job)


def run_inference_jobs_with_visible_devices(args) -> None:
    jobs = resolve_inference_jobs(args)
    visible_devices = parse_visible_devices()
    report_dir = resolve_report_dir(args)

    if len(visible_devices) <= 1:
        if visible_devices:
            print(f"Detected 1 visible GPU from CUDA_VISIBLE_DEVICES={visible_devices[0]}. Running in a single process.")
        else:
            print("No visible GPU detected from CUDA_VISIBLE_DEVICES. Running in a single process.")
        for job in jobs:
            run_single_inference_job(args, job)
        print_evaluation_summary(jobs, args.num_thresholds, report_dir=report_dir)
        return

    num_workers = len(visible_devices)
    job_buckets = split_jobs_across_workers(jobs, num_workers)
    print(
        f"Detected {num_workers} visible GPUs from CUDA_VISIBLE_DEVICES. "
        f"Launching {num_workers} worker processes."
    )

    ctx = mp.get_context("spawn")
    processes = []
    for worker_id, (visible_device, worker_jobs) in enumerate(zip(visible_devices, job_buckets)):
        process = ctx.Process(
            target=inference_worker,
            args=(worker_id, visible_device, worker_jobs, args),
        )
        process.start()
        processes.append(process)

    failed_workers = []
    for worker_id, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed_workers.append((worker_id, process.exitcode))

    if failed_workers:
        failures = ", ".join(f"worker {worker_id} exitcode={exitcode}" for worker_id, exitcode in failed_workers)
        raise RuntimeError(f"Inference workers failed: {failures}")

    print_evaluation_summary(jobs, args.num_thresholds, report_dir=report_dir)


def run_evaluate_jobs(args):
    jobs = resolve_evaluate_jobs(args)
    report_dir = resolve_report_dir(args)
    print_evaluation_summary(jobs, args.num_thresholds, report_dir=report_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified ProcessBench evaluation script supporting lm-head and value-head scoring.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["inference", "evaluate", "validate"],
    )
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=False)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset_names", type=str, nargs="+", default=None)

    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--scoring_mode", type=str, default="logp_ratio", choices=SCORING_MODES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--coef", type=float, default=COEF)
    parser.add_argument("--num_thresholds", type=int, default=1000)

    return parser.parse_args()


def main():
    global COEF

    args = parse_args()
    COEF = args.coef
    if args.tokenizer_path is None:
        tokenizer_path = args.model_path
        print(f"No --tokenizer_path specified. Using model_path as tokenizer_path: {tokenizer_path}")
        args.tokenizer_path = tokenizer_path

    if args.mode == "validate":
        args.mode = "evaluate"

    if args.mode == "inference":
        if not (args.model_path and args.tokenizer_path):
            raise ValueError("Please specify --model_path and --tokenizer_path for inference mode.")
        if args.scoring_mode == "logp_ratio" and not args.ref_model_path:
            raise ValueError("--ref_model_path is required when --scoring_mode=logp_ratio.")

        print(f"Using scoring mode: {args.scoring_mode}")
        run_inference_jobs_with_visible_devices(args)
    else:
        run_evaluate_jobs(args)


if __name__ == "__main__":
    main()

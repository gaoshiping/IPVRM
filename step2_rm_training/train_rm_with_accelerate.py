from __future__ import annotations

"""Compact Accelerate trainer for Step-2 reward-model training."""

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import normalize_prompt
from utils.accelerate import listify_token_ids, resolve_torch_dtype
from utils.datasets import load_parquet_dataset
from utils.tokenized_cache import build_data_fingerprint, build_tokenizer_fingerprint, load_or_create_tokenized_dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PAIRWISE_LOSSES = {"bt_sum", "bt_mean"}
POINTWISE_LOSSES = {"ipvrm", "implicit_prm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step-2 reward models with a minimal Accelerate loop.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--ref-model-name-or-path", type=str, default=None)
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--val-data", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-type", choices=["value_head", "logp", "logp_ratio"], required=True)
    parser.add_argument("--loss-type", choices=["bt_sum", "bt_mean", "ipvrm", "implicit_prm"], required=True)
    parser.add_argument("--chosen-key", type=str, default="chosen")
    parser.add_argument("--rejected-key", type=str, default="rejected")
    parser.add_argument("--conversation-key", type=str, default="conversation")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-val-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable model gradient checkpointing (enabled by default; use --no-gradient-checkpointing to disable).",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--logprob-chunk-size",
        type=int,
        default=128,
        help="Chunk size for chunk_Logp style token log-prob computation. Use 0 to disable chunking.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=16)

    parser.add_argument(
        "--tokenized-cache-root",
        type=Path,
        default=ROOT_DIR / ".cache" / "step2_tokenized",
        help="Directory used to save and reuse tokenized RM datasets.",
    )
    parser.add_argument("--overwrite-tokenized-cache", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def is_pairwise_loss(loss_type: str) -> bool:
    return loss_type in PAIRWISE_LOSSES


def empty_encoding() -> dict[str, Any]:
    return {
        "input_ids": [],
        "completion_mask": [],
        "target_tokens": 0,
    }


def encode_conversation(conversation: Any, tokenizer, max_length: int) -> dict[str, Any]:
    messages = normalize_prompt(conversation)
    if not messages or messages[-1]["role"] != "assistant":
        return empty_encoding()

    full_ids = listify_token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    prefix_ids = (
        listify_token_ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
        if len(messages) > 1
        else []
    )
    response_ids = full_ids[len(prefix_ids) :]
    if not response_ids:
        return empty_encoding()

    if len(response_ids) >= max_length:
        prompt_ids = []
        response_ids = response_ids[-max_length:]
    else:
        prompt_budget = max_length - len(response_ids)
        prompt_ids = prefix_ids[-prompt_budget:] if prompt_budget > 0 else []

    input_ids = prompt_ids + response_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "target_tokens": len(response_ids),
    }


def preprocess_pairwise_dataset(dataset: Dataset, tokenizer, args: argparse.Namespace) -> Dataset:
    def transform(example: dict[str, Any]) -> dict[str, Any]:
        chosen = encode_conversation(example[args.chosen_key], tokenizer, args.max_length)
        rejected = encode_conversation(example[args.rejected_key], tokenizer, args.max_length)
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_completion_mask": chosen["completion_mask"],
            "chosen_target_tokens": chosen["target_tokens"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_completion_mask": rejected["completion_mask"],
            "rejected_target_tokens": rejected["target_tokens"],
        }

    tokenized = dataset.map(transform, remove_columns=dataset.column_names, desc="Tokenizing pairwise RM data")
    return tokenized.filter(
        lambda row: row["chosen_target_tokens"] > 0 and row["rejected_target_tokens"] > 0,
        desc="Filtering empty pairwise rows",
    )


def preprocess_pointwise_dataset(dataset: Dataset, tokenizer, args: argparse.Namespace) -> Dataset:
    def transform(example: dict[str, Any]) -> dict[str, Any]:
        encoded = encode_conversation(example[args.conversation_key], tokenizer, args.max_length)
        return {
            "input_ids": encoded["input_ids"],
            "completion_mask": encoded["completion_mask"],
            "target_tokens": encoded["target_tokens"],
            "outcome_label": int(example[args.label_key]),
        }

    tokenized = dataset.map(transform, remove_columns=dataset.column_names, desc="Tokenizing pointwise RM data")
    return tokenized.filter(lambda row: row["target_tokens"] > 0, desc="Filtering empty pointwise rows")


def build_rm_tokenized_cache_payload(paths: list[str], tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tokenize_logic_version": 1,
        "data": build_data_fingerprint(paths),
        "loss_type": args.loss_type,
        "reward_type": args.reward_type,
        "max_length": args.max_length,
        "chosen_key": args.chosen_key,
        "rejected_key": args.rejected_key,
        "conversation_key": args.conversation_key,
        "label_key": args.label_key,
        **build_tokenizer_fingerprint(tokenizer),
    }


def load_rm_dataset(
    paths: list[str],
    tokenizer,
    args: argparse.Namespace,
    accelerator: Accelerator,
    split_name: str | None = None,
) -> tuple[Dataset, int | None, Path, str]:
    cache_payload = build_rm_tokenized_cache_payload(paths, tokenizer, args)

    def build_dataset() -> tuple[Dataset, int]:
        raw_dataset = load_parquet_dataset(paths)
        processed = (
            preprocess_pairwise_dataset(raw_dataset, tokenizer, args)
            if is_pairwise_loss(args.loss_type)
            else preprocess_pointwise_dataset(raw_dataset, tokenizer, args)
        )
        return processed, len(raw_dataset)

    return load_or_create_tokenized_dataset(
        accelerator=accelerator,
        cache_root=args.tokenized_cache_root,
        cache_payload=cache_payload,
        overwrite_cache=args.overwrite_tokenized_cache,
        build_dataset=build_dataset,
        split_name=split_name,
    )


def pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_length = max(len(sequence) for sequence in sequences)
    batch = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        if sequence:
            batch[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return batch


class PairwiseCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen_ids = [feature["chosen_input_ids"] for feature in features]
        rejected_ids = [feature["rejected_input_ids"] for feature in features]
        input_ids = chosen_ids + rejected_ids
        attention_mask = [[1] * len(ids) for ids in input_ids]
        completion_mask = (
            [feature["chosen_completion_mask"] for feature in features]
            + [feature["rejected_completion_mask"] for feature in features]
        )
        return {
            "input_ids": pad_sequences(input_ids, self.pad_token_id),
            "attention_mask": pad_sequences(attention_mask, 0),
            "completion_mask": pad_sequences(completion_mask, 0),
        }


class PointwiseCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        return {
            "input_ids": pad_sequences(input_ids, self.pad_token_id),
            "attention_mask": pad_sequences([[1] * len(ids) for ids in input_ids], 0),
            "completion_mask": pad_sequences([feature["completion_mask"] for feature in features], 0),
            "outcome_labels": torch.tensor([feature["outcome_label"] for feature in features], dtype=torch.float32),
        }


def gather_chunked_logprobs(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits to have shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels to have shape [batch, seq], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Shape mismatch between logits {tuple(logits.shape[:2])} and labels {tuple(labels.shape)}")

    sequence_length = logits.size(1)
    if sequence_length == 0:
        return logits.new_empty(labels.shape, dtype=torch.float32)

    def selective_log_softmax(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(chunk_logits.float(), dim=-1).gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)

    if chunk_size <= 0 or chunk_size >= sequence_length:
        return selective_log_softmax(logits, labels)

    chunks: list[torch.Tensor] = []
    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        chunks.append(selective_log_softmax(logits[:, start:end, :], labels[:, start:end]))
    return torch.cat(chunks, dim=1)


class RewardModel(nn.Module):
    def __init__(self, base_model: nn.Module, reward_type: str, logprob_chunk_size: int):
        super().__init__()
        self.base_model = base_model
        self.reward_type = reward_type
        self.logprob_chunk_size = logprob_chunk_size
        self.reward_head = None
        if reward_type == "value_head":
            self.reward_head = nn.Linear(self.hidden_size, 1, bias=False)
            self.reward_head.to(dtype=self._trainable_dtype)

    @property
    def hidden_size(self) -> int:
        config = self.base_model.config
        for attribute in ("hidden_size", "n_embd", "d_model"):
            if hasattr(config, attribute):
                return int(getattr(config, attribute))
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            for attribute in ("hidden_size", "n_embd", "d_model"):
                if hasattr(text_config, attribute):
                    return int(getattr(text_config, attribute))
        raise ValueError("Unable to infer hidden size for reward_head.")

    @property
    def _trainable_dtype(self) -> torch.dtype:
        for parameter in self.base_model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    def _compute_token_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_model: nn.Module | None = None,
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.reward_type == "value_head",
            use_cache=False,
            return_dict=True,
        )
        ensure_finite_tensor("policy.logits", outputs.logits)

        if self.reward_type == "value_head":
            hidden_states = outputs.hidden_states[-1][:, :-1, :]
            ensure_finite_tensor("policy.hidden_states_last", hidden_states)
            return self.reward_head(hidden_states).squeeze(-1)

        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        policy_logprobs = gather_chunked_logprobs(logits, labels, self.logprob_chunk_size)
        if self.reward_type == "logp":
            return policy_logprobs

        if ref_model is None:
            raise ValueError("logp_ratio reward requires a reference model.")

        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            ensure_finite_tensor("ref.logits", ref_outputs.logits)
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_logprobs = gather_chunked_logprobs(ref_logits, labels, self.logprob_chunk_size)
        return policy_logprobs - ref_logprobs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_model: nn.Module | None = None,
    ) -> torch.Tensor:
        return self._compute_token_rewards(input_ids, attention_mask, ref_model=ref_model)

    def forward_token_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_model: nn.Module | None = None,
    ) -> torch.Tensor:
        return self._compute_token_rewards(input_ids, attention_mask, ref_model=ref_model)


def reduce_sequence_rewards(token_rewards: torch.Tensor, completion_mask: torch.Tensor, reduction: str) -> torch.Tensor:
    masked_rewards = token_rewards * completion_mask
    if reduction == "sum":
        return masked_rewards.sum(dim=-1)
    return masked_rewards.sum(dim=-1) / completion_mask.sum(dim=-1).clamp_min(1.0)


def gather_last_valid(values: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    last_indices = completion_mask.long().sum(dim=-1).clamp_min(1) - 1
    return values.gather(1, last_indices.unsqueeze(-1)).squeeze(-1)


def build_prefix_values(
    token_rewards: torch.Tensor,
    completion_mask: torch.Tensor,
    reward_type: str,
    beta: float,
) -> torch.Tensor:
    if reward_type == "value_head":
        return token_rewards
    return beta * torch.cumsum(token_rewards * completion_mask, dim=-1)


def normalize_prefix_values(prefix_values: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    prefix_lengths = torch.cumsum(completion_mask, dim=-1).clamp_min(1.0)
    return prefix_values / prefix_lengths


def compute_ipvrm_margin_bce(
    normalized_prefix_values: torch.Tensor,
    expanded_labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    logits = normalized_prefix_values.float()
    labels = expanded_labels.float()
    return -(
        labels * F.logsigmoid(logits - margin)
        + (1.0 - labels) * F.logsigmoid(-(logits + margin))
    )


def reduce_mean_metrics(metrics: dict[str, torch.Tensor], accelerator: Accelerator) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for name, value in metrics.items():
        reduced[name] = accelerator.gather_for_metrics(value.detach().reshape(1)).mean().item()
    return reduced


def ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return

    total = tensor.numel()
    bad = total - int(finite_mask.sum().item())
    detached = tensor.detach().float()
    finite_values = detached[finite_mask]
    if finite_values.numel() > 0:
        finite_min = finite_values.min().item()
        finite_max = finite_values.max().item()
    else:
        finite_min = float("nan")
        finite_max = float("nan")
    raise FloatingPointError(
        f"Detected {bad}/{total} non-finite values in {name}; "
        f"finite_range=[{finite_min}, {finite_max}] shape={tuple(tensor.shape)}"
    )


def compute_pairwise_loss(
    model: RewardModel,
    ref_model: nn.Module | None,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Use model(...) so FSDP/DDP wrappers can run their forward hooks correctly.
    token_rewards = model(batch["input_ids"], batch["attention_mask"], ref_model=ref_model)
    ensure_finite_tensor("pairwise.token_rewards", token_rewards)
    completion_mask = batch["completion_mask"][:, 1:].float()
    reduction = "sum" if args.loss_type == "bt_sum" else "mean"
    sequence_rewards = reduce_sequence_rewards(token_rewards, completion_mask, reduction)
    ensure_finite_tensor("pairwise.sequence_rewards", sequence_rewards)

    pair_size = sequence_rewards.size(0) // 2
    chosen_rewards = sequence_rewards[:pair_size]
    rejected_rewards = sequence_rewards[pair_size:]
    logits = args.beta * (chosen_rewards - rejected_rewards) - args.gamma
    ensure_finite_tensor("pairwise.logits", logits)
    loss = F.softplus(-logits).mean()
    ensure_finite_tensor("pairwise.loss", loss.reshape(1))
    metrics = {
        "pair_accuracy": (logits > 0).float().mean(),
        "chosen_reward": chosen_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean(),
    }
    return loss, metrics


def compute_ipvrm_loss(
    model: RewardModel,
    ref_model: nn.Module | None,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_rewards = model(batch["input_ids"], batch["attention_mask"], ref_model=ref_model)
    ensure_finite_tensor("ipvrm.token_rewards", token_rewards)
    completion_mask = batch["completion_mask"][:, 1:].float()
    prefix_values = build_prefix_values(token_rewards, completion_mask, args.reward_type, args.beta)
    ensure_finite_tensor("ipvrm.prefix_values", prefix_values)
    normalized_prefix_values = normalize_prefix_values(prefix_values, completion_mask)
    ensure_finite_tensor("ipvrm.normalized_prefix_values", normalized_prefix_values)
    expanded_labels = batch["outcome_labels"].unsqueeze(-1).expand_as(normalized_prefix_values)
    margin = args.gamma

    # Eq. (9) in the paper applies BCE to the length-normalized prefix value
    # v_bar(t)=V_phi(s_t)/t with a symmetric margin m.
    per_token_loss = compute_ipvrm_margin_bce(normalized_prefix_values, expanded_labels, margin)
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)
    ensure_finite_tensor("ipvrm.loss", loss.reshape(1))

    predictions = (normalized_prefix_values > 0).float()
    accuracy = ((predictions == expanded_labels).float() * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)
    final_scores = gather_last_valid(normalized_prefix_values, completion_mask)
    positive_mask = batch["outcome_labels"] > 0.5
    negative_mask = ~positive_mask
    metrics = {
        "prefix_accuracy": accuracy,
        "positive_score": final_scores[positive_mask].mean() if positive_mask.any() else final_scores.new_tensor(0.0),
        "negative_score": final_scores[negative_mask].mean() if negative_mask.any() else final_scores.new_tensor(0.0),
    }
    return loss, metrics


def compute_implicit_prm_loss(
    model: RewardModel,
    ref_model: nn.Module | None,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_rewards = model(batch["input_ids"], batch["attention_mask"], ref_model=ref_model)
    ensure_finite_tensor("implicit_prm.token_rewards", token_rewards)
    completion_mask = batch["completion_mask"][:, 1:].float()

    if args.reward_type == "value_head":
        final_scores = gather_last_valid(token_rewards, completion_mask)
    else:
        final_scores = (token_rewards * completion_mask).sum(dim=-1)
    ensure_finite_tensor("implicit_prm.final_scores", final_scores)

    logits = args.beta * final_scores
    ensure_finite_tensor("implicit_prm.logits", logits)
    labels = batch["outcome_labels"]
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    ensure_finite_tensor("implicit_prm.loss", loss.reshape(1))

    predictions = (logits > 0).float()
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    metrics = {
        "implicit_prm_accuracy": (predictions == labels).float().mean(),
        "positive_score": final_scores[positive_mask].mean() if positive_mask.any() else final_scores.new_tensor(0.0),
        "negative_score": final_scores[negative_mask].mean() if negative_mask.any() else final_scores.new_tensor(0.0),
    }
    return loss, metrics


def compute_loss_and_metrics(
    model: RewardModel,
    ref_model: nn.Module | None,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if args.loss_type in PAIRWISE_LOSSES:
        return compute_pairwise_loss(model, ref_model, batch, args)
    if args.loss_type == "ipvrm":
        return compute_ipvrm_loss(model, ref_model, batch, args)
    return compute_implicit_prm_loss(model, ref_model, batch, args)


def evaluate(
    accelerator: Accelerator,
    model: RewardModel,
    ref_model: nn.Module | None,
    dataloader: DataLoader | None,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    if dataloader is None:
        return None

    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            loss, metrics = compute_loss_and_metrics(model, ref_model, batch, args)
            reduced = reduce_mean_metrics({"loss": loss, **metrics}, accelerator)
            for name, value in reduced.items():
                totals[name] = totals.get(name, 0.0) + value
            steps += 1

    model.train()
    if steps == 0:
        return None
    return {name: total / steps for name, total in totals.items()}


def build_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": resolve_torch_dtype(args.torch_dtype),
    }
    resolved_attn_impl = resolve_attn_implementation(args)
    if resolved_attn_impl:
        kwargs["attn_implementation"] = resolved_attn_impl
    return kwargs


def resolve_attn_implementation(args: argparse.Namespace) -> str | None:
    if args.attn_implementation:
        return args.attn_implementation

    model_name = str(args.model_name_or_path).lower()
    ref_model_name = str(args.ref_model_name_or_path).lower() if args.ref_model_name_or_path else ""
    tokenizer_name = str(args.tokenizer_name_or_path).lower() if args.tokenizer_name_or_path else ""
    combined = " ".join(part for part in (model_name, ref_model_name, tokenizer_name) if part)

    # Prefer FlashAttention-2 for Qwen3 when the kernel is available on a CUDA
    # machine. If it is unavailable, fall back to eager as the safer backend
    # for this FSDP+bf16 training setup.
    if "qwen3" in combined:
        if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "eager"
    return None


def load_causal_lm(model_name_or_path: str, **model_kwargs) -> nn.Module:
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except TypeError:
        if "dtype" not in model_kwargs:
            raise
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **fallback_kwargs)


def prepare_reference_model(accelerator: Accelerator, ref_model: nn.Module) -> nn.Module:
    # Keep the frozen reference model under the same distributed/mixed-precision
    # regime as the trainable policy model. A plain `.to(device)` reference model
    # has produced first-forward NaNs for Qwen3 under FSDP+bf16.
    ref_model = accelerator.prepare_model(ref_model)
    ref_model.requires_grad_(False)
    ref_model.eval()
    return ref_model


def enable_gradient_checkpointing(model: nn.Module) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def load_tokenizer(args: argparse.Namespace):
    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "use_fast": args.use_fast_tokenizer,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, fix_mistral_regex=True, **tokenizer_kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must provide either pad_token or eos_token.")
    return tokenizer


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_log_path(output_dir: Path) -> Path:
    return output_dir / "train.log"


def append_log(log_path: Path, message: str) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = str(message).splitlines() or [""]
    with log_path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"[{timestamp}] {line}\n")


def log_message(accelerator: Accelerator, log_path: Path, message: str, progress_bar=None) -> None:
    if accelerator.is_main_process:
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message)
        append_log(log_path, message)


def split_model_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base_state: dict[str, torch.Tensor] = {}
    reward_head_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key.startswith("base_model."):
            base_state[normalized_key[len("base_model.") :]] = value
        elif normalized_key.startswith("reward_head."):
            reward_head_state[normalized_key[len("reward_head.") :]] = value
    return base_state, reward_head_state


def save_checkpoint(
    accelerator: Accelerator,
    model: RewardModel,
    tokenizer,
    args: argparse.Namespace,
    tag: str,
) -> None:
    checkpoint_dir = args.output_dir / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_dict = accelerator.get_state_dict(model)
    base_state, reward_head_state = split_model_state_dict(state_dict)
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.base_model.save_pretrained(checkpoint_dir, state_dict=base_state, safe_serialization=True)
        generation_config = getattr(unwrapped_model.base_model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        if reward_head_state:
            torch.save(reward_head_state, checkpoint_dir / "reward_head.pt")
        save_json(
            checkpoint_dir / "rm_config.json",
            {
                "reward_type": args.reward_type,
                "loss_type": args.loss_type,
                "ref_model_name_or_path": args.ref_model_name_or_path,
                "beta": args.beta,
                "gamma": args.gamma,
                "max_length": args.max_length,
                "train_args": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
            },
        )
        append_log(get_log_path(args.output_dir), f"Saved checkpoint to {checkpoint_dir}")
    accelerator.wait_for_everyone()


def should_save_on_step(args: argparse.Namespace, global_step: int) -> bool:
    return args.save_strategy == "steps" and args.save_steps > 0 and global_step % args.save_steps == 0


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    args: argparse.Namespace,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def log_args(accelerator: Accelerator, log_path: Path, args: argparse.Namespace) -> None:
    log_message(accelerator, log_path, json.dumps(vars(args), ensure_ascii=False, indent=2, default=str))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = get_log_path(args.output_dir)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    log_message(accelerator, log_path, f"Starting RM training. Logs will be written to {log_path}")
    log_args(accelerator, log_path, args)
    resolved_attn_impl = resolve_attn_implementation(args)
    if args.attn_implementation is None and resolved_attn_impl is not None:
        log_message(accelerator, log_path, f"Auto-selected attn_implementation={resolved_attn_impl!r} for safer RM training.")

    tokenizer = load_tokenizer(args)
    train_dataset, train_raw_rows, train_cache_dir, train_cache_status = load_rm_dataset(
        args.train_data,
        tokenizer,
        args,
        accelerator,
        split_name="train",
    )
    if len(train_dataset) == 0:
        raise ValueError("No valid training rows remain after tokenization.")

    val_dataset = None
    val_raw_rows = None
    val_cache_dir = None
    val_cache_status = None
    if args.val_data:
        val_dataset, val_raw_rows, val_cache_dir, val_cache_status = load_rm_dataset(
            args.val_data,
            tokenizer,
            args,
            accelerator,
            split_name="val",
        )

    collator = PairwiseCollator(tokenizer.pad_token_id) if is_pairwise_loss(args.loss_type) else PointwiseCollator(
        tokenizer.pad_token_id
    )
    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        args=args,
    )
    val_dataloader = (
        build_dataloader(
            val_dataset,
            batch_size=args.per_device_val_batch_size,
            shuffle=False,
            collate_fn=collator,
            args=args,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    if train_raw_rows is not None:
        log_message(accelerator, log_path, f"Loaded {train_raw_rows} raw training rows and kept {len(train_dataset)} trainable rows.")
    else:
        log_message(accelerator, log_path, f"Loaded {len(train_dataset)} trainable training rows from cached tokenized data.")
    log_message(accelerator, log_path, f"Training tokenized cache {train_cache_status}: {train_cache_dir}")
    if val_dataset is not None and val_raw_rows is not None:
        log_message(
            accelerator,
            log_path,
            f"Loaded {val_raw_rows} raw validation rows and kept {len(val_dataset)} trainable rows.",
        )
    elif val_dataset is not None:
        log_message(
            accelerator,
            log_path,
            f"Loaded {len(val_dataset)} trainable validation rows from cached tokenized data.",
        )
    if val_dataset is not None and val_cache_dir is not None and val_cache_status is not None:
        log_message(accelerator, log_path, f"Validation tokenized cache {val_cache_status}: {val_cache_dir}")

    model_kwargs = build_model_kwargs(args)
    base_model = load_causal_lm(args.model_name_or_path, **model_kwargs)
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(base_model)
    model = RewardModel(base_model=base_model, reward_type=args.reward_type, logprob_chunk_size=args.logprob_chunk_size)

    ref_model = None
    if args.reward_type == "logp_ratio":
        ref_model_path = args.ref_model_name_or_path or args.model_name_or_path
        ref_model = load_causal_lm(ref_model_path, **model_kwargs)
        ref_model.config.use_cache = False
        ref_model.requires_grad_(False)
        ref_model.eval()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    prepare_items: list[Any] = [model, optimizer, train_dataloader]
    if val_dataloader is not None:
        prepare_items.append(val_dataloader)
    prepared_items = accelerator.prepare(*prepare_items)

    model = prepared_items[0]
    optimizer = prepared_items[1]
    train_dataloader = prepared_items[2]
    if val_dataloader is not None:
        val_dataloader = prepared_items[3]

    if ref_model is not None:
        log_message(accelerator, log_path, "Preparing frozen reference model with Accelerate for stable logp_ratio under FSDP.")
        ref_model = prepare_reference_model(accelerator, ref_model)

    num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warmup_steps = int(max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    global_step = 0
    running_loss = 0.0
    running_metrics: dict[str, float] = {}
    model.train()

    progress_bar = tqdm(
        total=max_train_steps,
        desc=f"Epoch 1/{args.num_train_epochs}",
        disable=not accelerator.is_main_process,
        leave=False,
        miniters=max(1, args.logging_steps),
    )

    for epoch in range(args.num_train_epochs):
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_train_epochs}")

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss, metrics = compute_loss_and_metrics(model, ref_model, batch, args)
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress_bar.update(1)
            reduced = reduce_mean_metrics({"loss": loss, **metrics}, accelerator)
            running_loss += reduced["loss"]
            for name, value in reduced.items():
                if name == "loss":
                    continue
                running_metrics[name] = running_metrics.get(name, 0.0) + value

            if global_step % args.logging_steps == 0:
                averaged_metrics = {name: value / args.logging_steps for name, value in running_metrics.items()}
                averaged_metrics["loss"] = running_loss / args.logging_steps
                averaged_metrics["lr"] = lr_scheduler.get_last_lr()[0]
                log_message(
                    accelerator,
                    log_path,
                    f"epoch={epoch + 1} step={global_step} metrics={json.dumps(averaged_metrics, ensure_ascii=False)}",
                    progress_bar=progress_bar,
                )
                running_loss = 0.0
                running_metrics = {}

            if should_save_on_step(args, global_step):
                save_checkpoint(accelerator, model, tokenizer, args, f"checkpoint-step-{global_step}")

        eval_metrics = evaluate(accelerator, model, ref_model, val_dataloader, args)
        if eval_metrics is not None:
            log_message(
                accelerator,
                log_path,
                f"epoch={epoch + 1} eval={json.dumps(eval_metrics, ensure_ascii=False)}",
                progress_bar=progress_bar,
            )

        if args.save_strategy == "epoch":
            save_checkpoint(accelerator, model, tokenizer, args, f"checkpoint-epoch-{epoch + 1}")

    progress_bar.close()

    save_checkpoint(accelerator, model, tokenizer, args, "final")
    log_message(accelerator, log_path, "Training finished successfully.")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Stage-1 supervised fine-tuning that initializes the policy before RM and DistRL."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from datasets import Dataset
from safetensors.torch import save_model as save_safetensors_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.accelerate import listify_token_ids, resolve_torch_dtype
from utils.datasets import load_parquet_dataset
from utils.tokenized_cache import build_data_fingerprint, build_tokenizer_fingerprint, load_or_create_tokenized_dataset


def format_seconds(total_seconds: float) -> str:
    """Format seconds into HH:MM:SS for progress logging."""
    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Stage-1 SFT run."""
    parser = argparse.ArgumentParser(
        description="Standalone chat SFT trainer based on Hugging Face Accelerate. Supervises only the final assistant reply."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        nargs="+",
        required=True,
        help="One or more parquet files containing processed conversations.",
    )
    parser.add_argument(
        "--conversation-key",
        type=str,
        default="conversations",
        help="Column name that stores the chat conversation list.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Pretrained model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help="Optional tokenizer path. Defaults to the model path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used to save checkpoints and the final model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=3072,
        help="Maximum number of tokens kept for each training example. Left truncation is used to preserve the final assistant reply.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Per-device micro batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Linear warmup ratio.")
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--preprocessing-num-workers",
        type=int,
        default=16,
        help="Number of workers used for tokenization.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N optimizer steps.")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Save a checkpoint every N optimizer steps. Use 0 to disable intermediate checkpointing.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable model gradient checkpointing (enabled by default; use --no-gradient-checkpointing to disable).",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attention implementation, for example flash_attention_2.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote code when loading the model and tokenizer.",
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="Use the slow tokenizer implementation.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--tokenized-cache-root",
        type=Path,
        default=ROOT_DIR / ".cache" / "step1_tokenized",
        help="Directory used to save and reuse tokenized SFT datasets.",
    )
    parser.add_argument(
        "--overwrite-tokenized-cache",
        action="store_true",
        help="Ignore any saved tokenized dataset and rebuild it from raw parquet files.",
    )
    return parser.parse_args()


def tokenize_conversation(example: dict[str, Any], tokenizer, conversation_key: str, max_length: int) -> dict[str, Any]:
    """Tokenize one conversation and supervise only the final assistant reply.

    This matches the paper's initialization stage, where SFT produces an initial
    policy that is later reused for rollout sampling and RM construction.
    """
    conversation = example.get(conversation_key)
    if not isinstance(conversation, list) or not conversation:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_mask": [],
            "target_tokens": 0,
            "sequence_length": 0,
        }

    if conversation[-1].get("role") != "assistant":
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_mask": [],
            "target_tokens": 0,
            "sequence_length": 0,
        }

    full_ids = listify_token_ids(
        tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_ids = []
    elif isinstance(eos_token_id, int):
        eos_token_ids = [eos_token_id]
    else:
        eos_token_ids = listify_token_ids(eos_token_id)
    if eos_token_ids and full_ids[-len(eos_token_ids) :] != eos_token_ids:
        full_ids = full_ids + eos_token_ids

    prefix_ids = listify_token_ids(
        tokenizer.apply_chat_template(
            conversation[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    prefix_length = len(prefix_ids)
    if len(full_ids) > max_length:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_mask": [],
            "target_tokens": 0,
            "sequence_length": len(full_ids),
        }

    attention_mask = [1] * len(full_ids)
    loss_mask = [0] * len(full_ids)
    for index in range(prefix_length, len(full_ids)):
        loss_mask[index] = 1

    if len(full_ids) != len(loss_mask):
        raise ValueError("Token ids and loss mask must have the same length for SFT supervision.")
    labels = [token_id if mask == 1 else -100 for token_id, mask in zip(full_ids, loss_mask)]
    target_tokens = sum(loss_mask)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
        "target_tokens": target_tokens,
        "sequence_length": len(full_ids),
    }


def build_sft_tokenized_cache_payload(args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    """Describe the Stage-1 preprocessing config used to validate cache reuse."""
    return {
        "tokenize_logic_version": 2,
        "train_data": build_data_fingerprint(args.train_data),
        "conversation_key": args.conversation_key,
        "max_length": args.max_length,
        "max_train_samples": args.max_train_samples,
        **build_tokenizer_fingerprint(tokenizer),
    }


def build_sft_tokenized_dataset(
    args: argparse.Namespace,
    tokenizer,
    raw_dataset: Dataset | None = None,
) -> tuple[Dataset, int]:
    """Build the Stage-1 tokenized dataset from raw parquet data."""
    if raw_dataset is None:
        raw_dataset = load_parquet_dataset(args.train_data)
        if args.max_train_samples is not None:
            raw_dataset = raw_dataset.select(range(min(args.max_train_samples, len(raw_dataset))))

    tokenize_fn = lambda example: tokenize_conversation(
        example=example,
        tokenizer=tokenizer,
        conversation_key=args.conversation_key,
        max_length=args.max_length,
    )

    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing SFT conversations",
    )
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: example["target_tokens"] > 0 and example["sequence_length"] <= args.max_length,
        num_proc=max(1, min(args.preprocessing_num_workers, os.cpu_count() or 1)),
        desc="Filtering samples without supervised assistant tokens or above max length",
    )
    return tokenized_dataset, len(raw_dataset)


def load_or_create_sft_tokenized_dataset(
    accelerator: Accelerator,
    args: argparse.Namespace,
    tokenizer,
) -> tuple[Dataset, int | None, Path, str]:
    """Reuse or build the cached Stage-1 tokenized dataset."""
    return load_or_create_tokenized_dataset(
        accelerator=accelerator,
        cache_root=args.tokenized_cache_root,
        cache_payload=build_sft_tokenized_cache_payload(args, tokenizer),
        overwrite_cache=args.overwrite_tokenized_cache,
        build_dataset=lambda: build_sft_tokenized_dataset(args, tokenizer),
    )


class ConversationOnlyLastReplyCollator:
    """Pad Stage-1 SFT examples while preserving final-reply-only supervision."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Right-pad variable-length conversations into a dense training batch."""
        max_length = max(len(feature["input_ids"]) for feature in features)

        input_ids = []
        attention_mask = []
        labels = []
        loss_mask = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)
            loss_mask.append(feature["loss_mask"] + [0] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }


def save_model_checkpoint(
    accelerator: Accelerator,
    model,
    tokenizer,
    output_dir: Path,
    training_config: dict[str, Any],
) -> None:
    """Save the initialized SFT policy for later RM and RL stages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.distributed_type == DistributedType.NO:
        model.config.save_pretrained(output_dir)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output_dir)
        save_safetensors_model(model, str(output_dir / "model.safetensors"), metadata={"format": "pt"})
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
                json.dump(training_config, file, ensure_ascii=False, indent=2)
        return

    state_dict = accelerator.get_state_dict(model)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
            json.dump(training_config, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run Stage-1 SFT to obtain the initial policy used by later stages."""
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.use_slow_tokenizer,
    )
    if tokenizer.chat_template is None:
        raise ValueError("The tokenizer does not define a chat template, so apply_chat_template cannot be used.")

    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    torch_dtype = resolve_torch_dtype(args.torch_dtype, auto_value=None)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    tokenized_dataset, raw_dataset_rows, tokenized_cache_dir, cache_status = load_or_create_sft_tokenized_dataset(
        accelerator=accelerator,
        args=args,
        tokenizer=tokenizer,
    )
    tokenized_dataset = tokenized_dataset.shuffle(seed=args.seed)

    if len(tokenized_dataset) == 0:
        raise ValueError("No valid training samples remain after tokenization. Check conversation_key and max_length.")

    if accelerator.is_main_process:
        avg_sequence_length = sum(tokenized_dataset["sequence_length"]) / len(tokenized_dataset)
        avg_target_tokens = sum(tokenized_dataset["target_tokens"]) / len(tokenized_dataset)
        if raw_dataset_rows is not None:
            print(f"Loaded {raw_dataset_rows} raw rows and kept {len(tokenized_dataset)} trainable rows.")
        else:
            print(f"Loaded {len(tokenized_dataset)} trainable rows from cached tokenized data.")
        print(f"Tokenized dataset cache {cache_status}: {tokenized_cache_dir}")
        print(f"Average sequence length: {avg_sequence_length:.2f}")
        print(f"Average supervised tokens: {avg_target_tokens:.2f}")

    collator = ConversationOnlyLastReplyCollator(pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        collate_fn=collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    # Compute scheduler steps from the per-rank dataloader after prepare().
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warmup_steps = int(max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    training_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    training_config["train_data"] = [str(path) for path in args.train_data]
    training_config["output_dir"] = str(args.output_dir)
    training_config["world_size"] = accelerator.num_processes
    training_config["mixed_precision"] = accelerator.mixed_precision
    training_config["distributed_type"] = str(accelerator.distributed_type)

    completed_steps = 0
    running_loss = 0.0
    running_target_tokens = 0
    running_micro_steps = 0
    train_start_time = time.time()

    for epoch in range(args.num_train_epochs):
        model.train()
        update_bar = tqdm(
            total=num_update_steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
            disable=not accelerator.is_main_process,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            gathered_loss = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
            gathered_target_tokens = accelerator.gather(batch["loss_mask"].sum().detach().unsqueeze(0)).sum().item()
            running_loss += gathered_loss
            running_target_tokens += gathered_target_tokens
            running_micro_steps += 1

            if accelerator.sync_gradients:
                completed_steps += 1
                update_bar.update(1)

                if completed_steps % args.logging_steps == 0:
                    avg_loss = running_loss / max(running_micro_steps, 1)
                    avg_target_tokens = running_target_tokens / max(running_micro_steps, 1)
                    current_lr = optimizer.param_groups[0]["lr"]
                    local_max_seq_len = batch["attention_mask"].sum(dim=1).max().detach()
                    max_seq_len = int(accelerator.gather_for_metrics(local_max_seq_len.unsqueeze(0)).max().item())
                    if accelerator.is_main_process:
                        update_bar.write(
                            f"epoch={epoch + 1} step={completed_steps}/{max_train_steps} "
                            f"loss={avg_loss:.4f} lr={current_lr:.6e} target_tokens={avg_target_tokens:.2f} "
                            f"max_seq_len={max_seq_len}"
                        )
                    running_loss = 0.0
                    running_target_tokens = 0
                    running_micro_steps = 0

                if args.save_steps > 0 and completed_steps % args.save_steps == 0:
                    save_model_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=args.output_dir / f"checkpoint-step-{completed_steps}",
                        training_config=training_config,
                    )
        update_bar.close()

    save_model_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir / "final",
        training_config=training_config,
    )

    if accelerator.is_main_process:
        print(f"Training finished. Final checkpoint saved to: {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()

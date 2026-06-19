from __future__ import annotations

import argparse
import os

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_tiny_loader(batch_size: int) -> DataLoader:
    inputs = torch.randn(16, 1024)
    labels = torch.randint(0, 4, (16,))
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal distributed repro for DDP/FSDP crashes.")
    parser.add_argument("--mode", choices=["tiny", "hf"], required=True)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
    accelerator.print(f"rank={rank} stage=init mode={args.mode}")

    if args.mode == "tiny":
        model = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 4))
        optimizer = AdamW(model.parameters(), lr=1e-3)
        dataloader = build_tiny_loader(args.batch_size)
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        accelerator.print(f"rank={rank} stage=prepared")
        batch_inputs, batch_labels = next(iter(dataloader))
        loss = nn.functional.cross_entropy(model(batch_inputs), batch_labels)
    else:
        if not args.model_name_or_path:
            raise ValueError("--model-name-or-path is required for --mode hf")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        encoded = tokenizer(["What is 2+2?"] * 4, return_tensors="pt", padding=True)
        labels = encoded["input_ids"].clone()
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model.config.use_cache = False
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        accelerator.print(f"rank={rank} stage=prepared")
        input_ids, attention_mask, labels = next(iter(dataloader))
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    accelerator.print(f"rank={rank} stage=step_done loss={loss.item():.6f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

from __future__ import annotations

from .datasets import load_dataset, load_jsonl, load_single_dataset, save_dataset


def split_batch(prompts: list[str], batch_size: int) -> list[list[str]]:
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]


__all__ = [
    "load_dataset",
    "load_jsonl",
    "load_single_dataset",
    "save_dataset",
    "split_batch",
]

from __future__ import annotations

"""Shared helpers for caching tokenized Hugging Face datasets across training runs."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from accelerate import Accelerator
from datasets import Dataset, load_from_disk


TOKENIZED_DATASET_CACHE_VERSION = 1


def build_data_fingerprint(paths: list[str]) -> list[dict[str, Any]]:
    """Fingerprint local dataset files so cache reuse follows file changes."""
    fingerprint = []
    for path_str in paths:
        path = Path(path_str).expanduser().resolve()
        stat = path.stat()
        fingerprint.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprint


def build_tokenizer_fingerprint(tokenizer) -> dict[str, Any]:
    """Fingerprint tokenizer settings that affect tokenized dataset contents."""
    return {
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template": tokenizer.chat_template,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }


def build_tokenized_dataset_cache_dir(
    cache_root: Path,
    cache_payload: dict[str, Any],
    split_name: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Derive a stable cache directory from a JSON-serializable preprocessing payload."""
    normalized_payload = {"version": TOKENIZED_DATASET_CACHE_VERSION, **cache_payload}
    cache_key = hashlib.sha256(json.dumps(normalized_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if split_name is None:
        return cache_root / cache_key, normalized_payload
    return cache_root / split_name / cache_key, normalized_payload


def load_tokenized_cache_metadata(cache_dir: Path) -> dict[str, Any] | None:
    """Read cache metadata written next to a saved tokenized dataset."""
    metadata_path = cache_dir / "cache_meta.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_tokenized_cache_metadata(cache_dir: Path, metadata: dict[str, Any]) -> None:
    """Persist metadata so later runs can report what cache was reused."""
    metadata_path = cache_dir / "cache_meta.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_or_create_tokenized_dataset(
    accelerator: Accelerator,
    cache_root: Path,
    cache_payload: dict[str, Any],
    overwrite_cache: bool,
    build_dataset: Callable[[], tuple[Dataset, int]],
    split_name: str | None = None,
) -> tuple[Dataset, int | None, Path, str]:
    """Reuse a saved tokenized dataset when possible, otherwise build it once."""
    cache_dir, normalized_payload = build_tokenized_dataset_cache_dir(
        cache_root=cache_root,
        cache_payload=cache_payload,
        split_name=split_name,
    )
    raw_dataset_rows: int | None = None
    cache_status = "created"
    tokenized_dataset: Dataset | None = None

    with accelerator.main_process_first():
        if accelerator.is_main_process:
            cache_reused = cache_dir.exists() and not overwrite_cache
            if cache_reused:
                try:
                    tokenized_dataset = load_from_disk(str(cache_dir))
                    cache_status = "loaded"
                    cache_metadata = load_tokenized_cache_metadata(cache_dir)
                    if cache_metadata is not None:
                        raw_dataset_rows = cache_metadata.get("raw_dataset_rows")
                except Exception:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    cache_reused = False

            if not cache_reused:
                tokenized_dataset, raw_dataset_rows = build_dataset()
                cache_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(cache_dir, ignore_errors=True)
                tokenized_dataset.save_to_disk(str(cache_dir))
                save_tokenized_cache_metadata(
                    cache_dir,
                    {
                        "cache_payload": normalized_payload,
                        "raw_dataset_rows": raw_dataset_rows,
                        "tokenized_rows": len(tokenized_dataset),
                    },
                )

    if tokenized_dataset is None:
        tokenized_dataset = load_from_disk(str(cache_dir))

    return tokenized_dataset, raw_dataset_rows, cache_dir, cache_status
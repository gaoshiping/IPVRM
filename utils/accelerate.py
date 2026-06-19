from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def resolve_torch_dtype(
    dtype_name: str,
    *,
    auto_value: torch.dtype | str | None = "auto",
) -> torch.dtype | str | None:
    mapping: dict[str, torch.dtype | str | None] = {
        "auto": auto_value,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from error


def _unwrap_token_ids_container(token_ids: Any) -> Any:
    # Newer tokenizer.apply_chat_template() calls can return a BatchEncoding
    # mapping instead of a plain list of token ids.
    if isinstance(token_ids, Mapping):
        if "input_ids" not in token_ids:
            raise TypeError(f"Token mapping must contain 'input_ids', got keys: {list(token_ids.keys())}")
        token_ids = token_ids["input_ids"]

    if hasattr(token_ids, "tolist") and not isinstance(token_ids, list):
        converted = token_ids.tolist()
        if converted is not token_ids:
            token_ids = converted

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError(f"Expected a single token sequence, got a batch of {len(token_ids)} sequences.")
        token_ids = token_ids[0]

    return token_ids


def listify_token_ids(token_ids: Any) -> list[int]:
    token_ids = _unwrap_token_ids_container(token_ids)
    if isinstance(token_ids, list):
        return [int(token_id) for token_id in token_ids]
    try:
        return [int(token_id) for token_id in token_ids]
    except TypeError as error:
        raise TypeError(f"Unsupported token container type: {type(token_ids)}") from error

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.chat import append_assistant_message, canonical_prompt, derive_prompt_id, normalize_message, normalize_prompt
from utils.datasets import load_records, preview_records, save_records_as_parquet

__all__ = [
    "append_assistant_message",
    "canonical_prompt",
    "derive_prompt_id",
    "load_records",
    "normalize_message",
    "normalize_prompt",
    "preview_records",
    "save_records_as_parquet",
]
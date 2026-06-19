from __future__ import annotations

"""Stage-1 data preparation for the SFT policy initialization pipeline."""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.chat import normalize_role


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Eurus-2 SFT preprocessing step."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess PRIME-RL/Eurus-2-SFT-Data into a role/content conversation parquet file."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PRIME-RL/Eurus-2-SFT-Data",
        help="Hugging Face dataset name.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download.")
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output parquet file path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=16,
        help="Number of processes used during dataset mapping.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    return parser.parse_args()


def convert_example(example: dict) -> dict:
    """Convert one math Eurus record into the conversation-only format used for Stage-1 SFT."""
    processed_conversations = []
    system_text = example.get("system")
    if isinstance(system_text, str) and system_text.strip():
        processed_conversations.append({"role": "system", "content": system_text})

    for message in example["conversations"]:
        processed_conversations.append(
            {
                "role": normalize_role(message["from"]),
                "content": message["value"],
            }
        )

    return {"conversations": processed_conversations}


def main() -> None:
    """Download, normalize, and export Eurus-2 data for the initial SFT policy."""
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {args.dataset_name} [{args.split}]")
    dataset = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    print(f"Downloaded {len(dataset)} rows.")

    dataset = dataset.filter(
        lambda example: str(example.get("task", "")).lower() == "math",
        num_proc=args.num_proc,
        desc="Filtering math task samples",
    )
    print(f"Kept {len(dataset)} math rows.")

    if args.max_samples is not None:
        capped_size = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(capped_size))
        print(f"Using the first {len(dataset)} math rows for preprocessing.")

    processed_dataset = dataset.map(
        convert_example,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Converting Eurus conversations to role/content format",
    )

    processed_dataset.to_parquet(str(args.output_file))
    print(f"Saved processed parquet to: {args.output_file}")

    preview_count = min(3, len(processed_dataset))
    print("\nPreview samples:")
    for index in range(preview_count):
        print(f"\n===== Sample {index + 1} =====")
        print(json.dumps(processed_dataset[index], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
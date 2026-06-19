from __future__ import annotations

"""Stage-2 data preparation for reward-model training data selection."""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import load_dataset
from utils import save_dataset

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the DAPO-Math-17k preprocessing step."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess open-r1/DAPO-Math-17k-Processed, extract source_prompt as prompt and keep reward_model."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="open-r1/DAPO-Math-17k-Processed",
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
    """Keep only the fields required by RM stage data construction and rename source_prompt to prompt."""
    return {
        "data_source": example.get("data_source", "math_dapo"),
        "prompt": example.get("source_prompt"),
        "reward_model": example.get("reward_model"),
    }


def main() -> None:
    """Download, filter, and export DAPO-Math data for RM stage."""
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {args.dataset_name} [{args.split}]")
    dataset = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    print(f"Downloaded {len(dataset)} rows.")

    if args.max_samples is not None:
        capped_size = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(capped_size))
        print(f"Using the first {len(dataset)} rows for preprocessing.")

    processed_dataset = dataset.map(
        convert_example,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Mapping source_prompt to prompt and keeping reward_model",
    )

    save_dataset(processed_dataset, args.output_file)
    print(f"Saved processed dataset to: {args.output_file}")

    preview_count = min(3, len(processed_dataset))
    print("\nPreview samples:")
    for index in range(preview_count):
        print(f"\n===== Sample {index + 1} =====")
        print(json.dumps(processed_dataset[index], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
python step2_rm_training/preprocess_dapo17k.py --output-file step2_rm_training/data/dapo_math_processed.parquet
"""

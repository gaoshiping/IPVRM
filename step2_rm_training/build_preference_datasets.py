from __future__ import annotations

"""Build minimal pairwise and pointwise RM datasets from scored rollouts."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value

from common import append_assistant_message, derive_prompt_id, load_records, normalize_prompt, preview_records, save_records_as_parquet


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building RM datasets from scored rollouts."""
    parser = argparse.ArgumentParser(description="Build minimal pairwise and pointwise RM datasets from scored rollouts.")
    parser.add_argument("--input-path", type=str, required=True, help="Scored rollout parquet path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory used to save RM datasets.")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Prompt column name.")
    parser.add_argument("--prompt-id-key", type=str, default="prompt_id", help="Prompt id column name.")
    parser.add_argument("--response-key", type=str, default="responses", help="Responses column name.")
    parser.add_argument("--score-key", type=str, default="score", help="Per-response correctness column name.")
    parser.add_argument(
        "--pair-strategy",
        choices=["best_worst", "cartesian", "random"],
        default="best_worst",
        help="How chosen and rejected responses are paired within each prompt.",
    )
    parser.add_argument(
        "--max-pairs-per-prompt",
        type=int,
        default=1,
        help="Maximum number of chosen/rejected pairs kept for each prompt. Use 0 for no limit.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.95, help="Prompt-level train split ratio.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    return parser.parse_args()


def ensure_list(value: Any) -> list[Any]:
    """Normalize a possibly scalar field to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def split_prompt_ids(prompt_ids: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    """Split prompt ids to avoid prompt leakage across train and validation."""
    prompt_ids = sorted(set(prompt_ids))
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)

    if len(prompt_ids) <= 1:
        return set(prompt_ids), set()

    train_count = round(len(prompt_ids) * train_ratio)
    train_count = min(max(train_count, 1), len(prompt_ids) - 1)
    train_prompt_ids = set(prompt_ids[:train_count])
    val_prompt_ids = set(prompt_ids[train_count:])
    return train_prompt_ids, val_prompt_ids


def save_split_records(
    records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    output_path: Path,
    empty_features: Features | None = None,
) -> None:
    """Write a split to parquet while preserving schema even for empty splits."""
    if records:
        save_records_as_parquet(records, output_path)
        return
    if reference_records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features = Dataset.from_list(reference_records).features
        empty_dataset = Dataset.from_list([], features=features)
        empty_dataset.to_parquet(str(output_path))
        return
    if empty_features is None:
        raise ValueError(f"Cannot write empty split without reference schema: {output_path}")
    save_records_or_empty([], output_path, empty_features)


def save_records_or_empty(records: list[dict[str, Any]], output_path: Path, features: Features) -> None:
    """Write records to parquet, preserving schema even when the dataset is empty."""
    if records:
        save_records_as_parquet(records, output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    empty_dataset = Dataset.from_list([], features=features)
    empty_dataset.to_parquet(str(output_path))


def select_pairs(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    pair_strategy: str,
    max_pairs_per_prompt: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select positive/negative trajectory pairs from prompts with mixed outcomes."""
    pair_limit = None if max_pairs_per_prompt <= 0 else max_pairs_per_prompt
    positives = sorted(positives, key=lambda item: item["response_index"])
    negatives = sorted(negatives, key=lambda item: item["response_index"])

    if pair_strategy == "best_worst":
        pair_count = min(len(positives), len(negatives))
        if pair_limit is not None:
            pair_count = min(pair_count, pair_limit)
        return list(zip(positives[:pair_count], negatives[:pair_count]))

    if pair_strategy == "random":
        positives = positives[:]
        negatives = negatives[:]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        pair_count = min(len(positives), len(negatives))
        if pair_limit is not None:
            pair_count = min(pair_count, pair_limit)
        return list(zip(positives[:pair_count], negatives[:pair_count]))

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for positive in positives:
        for negative in negatives:
            pairs.append((positive, negative))
            if pair_limit is not None and len(pairs) >= pair_limit:
                return pairs
    return pairs


def filter_by_prompt_ids(records: list[dict[str, Any]], allowed_prompt_ids: set[str]) -> list[dict[str, Any]]:
    """Keep only records whose internal prompt id belongs to the target split."""
    return [record for record in records if record["_prompt_id"] in allowed_prompt_ids]


def strip_internal_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop helper metadata before saving the public parquet schema."""
    return [{key: value for key, value in record.items() if not key.startswith("_")} for record in records]


def build_pointwise_from_pairwise(pairwise_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each chosen/rejected pair into two pointwise examples."""
    pointwise_records: list[dict[str, Any]] = []
    for record in pairwise_records:
        pointwise_records.append(
            {
                "_prompt_id": record["_prompt_id"],
                "conversation": record["chosen"],
                "label": 1,
            }
        )
        pointwise_records.append(
            {
                "_prompt_id": record["_prompt_id"],
                "conversation": record["rejected"],
                "label": 0,
            }
        )
    return pointwise_records


def main() -> None:
    """Build minimal pairwise and pointwise RM datasets from verifier-scored rollouts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_records = load_records(args.input_path)
    rng = random.Random(args.seed)

    pairwise_records: list[dict[str, Any]] = []
    kept_prompt_ids: list[str] = []
    kept_prompts = 0
    skipped_prompts = 0

    for record in scored_records:
        prompt = normalize_prompt(record[args.prompt_key])
        prompt_id = derive_prompt_id(record, args.prompt_key, args.prompt_id_key)
        responses = ensure_list(record.get(args.response_key))
        scores = ensure_list(record.get(args.score_key))

        if len(responses) != len(scores):
            raise ValueError("responses and score must share the same length.")

        candidates: list[dict[str, Any]] = []
        for response_index, (response, score) in enumerate(zip(responses, scores)):
            label = int(score)
            candidates.append(
                {
                    "response_index": response_index,
                    "label": label,
                    "conversation": append_assistant_message(prompt, str(response)),
                }
            )

        positives = [candidate for candidate in candidates if candidate["label"] == 1]
        negatives = [candidate for candidate in candidates if candidate["label"] == 0]
        if not positives or not negatives:
            skipped_prompts += 1
            continue

        selected_pairs = select_pairs(
            positives=positives,
            negatives=negatives,
            pair_strategy=args.pair_strategy,
            max_pairs_per_prompt=args.max_pairs_per_prompt,
            rng=rng,
        )
        if not selected_pairs:
            skipped_prompts += 1
            continue

        kept_prompts += 1
        kept_prompt_ids.append(prompt_id)

        for chosen, rejected in selected_pairs:
            pairwise_records.append(
                {
                    "_prompt_id": prompt_id,
                    "chosen": chosen["conversation"],
                    "rejected": rejected["conversation"],
                }
            )

    train_prompt_ids, val_prompt_ids = split_prompt_ids(kept_prompt_ids, args.train_ratio, args.seed)
    pointwise_records = build_pointwise_from_pairwise(pairwise_records)

    pairwise_features = Features(
        {
            "chosen": [{"role": Value("string"), "content": Value("string")}],
            "rejected": [{"role": Value("string"), "content": Value("string")}],
        }
    )
    pointwise_features = Features(
        {
            "conversation": [{"role": Value("string"), "content": Value("string")}],
            "label": Value("int64"),
        }
    )

    pairwise_all = strip_internal_fields(pairwise_records)
    pointwise_all = strip_internal_fields(pointwise_records)
    pairwise_train = strip_internal_fields(filter_by_prompt_ids(pairwise_records, train_prompt_ids))
    pairwise_val = strip_internal_fields(filter_by_prompt_ids(pairwise_records, val_prompt_ids))
    pointwise_train = strip_internal_fields(filter_by_prompt_ids(pointwise_records, train_prompt_ids))
    pointwise_val = strip_internal_fields(filter_by_prompt_ids(pointwise_records, val_prompt_ids))

    save_records_or_empty(pairwise_all, args.output_dir / "pairwise_all.parquet", pairwise_features)
    save_records_or_empty(pointwise_all, args.output_dir / "pointwise_all.parquet", pointwise_features)
    save_split_records(pairwise_train, pairwise_all, args.output_dir / "pairwise_train.parquet", pairwise_features)
    save_split_records(pairwise_val, pairwise_all, args.output_dir / "pairwise_val.parquet", pairwise_features)
    save_split_records(pointwise_train, pointwise_all, args.output_dir / "pointwise_train.parquet", pointwise_features)
    save_split_records(pointwise_val, pointwise_all, args.output_dir / "pointwise_val.parquet", pointwise_features)

    summary = {
        "scored_prompt_count": len(scored_records),
        "kept_prompts": kept_prompts,
        "skipped_prompts": skipped_prompts,
        "pairwise_all_count": len(pairwise_all),
        "pairwise_train_count": len(pairwise_train),
        "pairwise_val_count": len(pairwise_val),
        "pointwise_all_count": len(pointwise_all),
        "pointwise_train_count": len(pointwise_train),
        "pointwise_val_count": len(pointwise_val),
        "pair_strategy": args.pair_strategy,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    preview_records(pairwise_all)
    preview_records(pointwise_all)


if __name__ == "__main__":
    main()

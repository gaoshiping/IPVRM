from __future__ import annotations

"""Score rollout trajectories with the verifier path used by step3 trainer validation."""

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from common import load_records, save_records_as_parquet
from verl.utils.reward_score import default_compute_score


@dataclass(frozen=True)
class ScoreTask:
    record_index: int
    response_index: int
    data_source: str
    model_output: str
    ground_truth: str
    extra_info: Any
    timeout_seconds: float
    label_threshold: float


@dataclass(frozen=True)
class ScoreResult:
    record_index: int
    response_index: int
    label: int
    raw_score: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score rollout responses with trainer validation's default_compute_score path."
    )
    parser.add_argument("--input-path", type=str, required=True, help="Rollout parquet path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Scored parquet path.")
    parser.add_argument("--response-key", type=str, default="responses", help="Responses column name.")
    parser.add_argument("--reward-model-key", type=str, default="reward_model", help="Reward model metadata column.")
    parser.add_argument("--data-source-key", type=str, default="data_source", help="Data source column name.")
    parser.add_argument("--extra-info-key", type=str, default="extra_info", help="Extra info column name.")
    parser.add_argument("--score-key", type=str, default="trainer_verify_score", help="Output label field.")
    parser.add_argument(
        "--raw-score-key",
        type=str,
        default="trainer_verify_raw_score",
        help="Output raw score field. Use empty string to disable.",
    )
    parser.add_argument(
        "--status-key",
        type=str,
        default="trainer_verify_status",
        help="Output status field. Use empty string to disable.",
    )
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Accepted for CLI compatibility; trainer verifier uses its own internal timeout.",
    )
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=32)
    return parser.parse_args()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def score_to_float(result: Any) -> float:
    if isinstance(result, dict):
        for key in ("acc", "score"):
            if key in result:
                return float(result[key])
        for value in result.values():
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"No scalar value found in reward score dict keys={list(result.keys())}")
    return float(result)


def build_score_tasks(
    records: list[dict[str, Any]],
    response_key: str,
    reward_model_key: str,
    data_source_key: str,
    extra_info_key: str,
    timeout_seconds: float,
    label_threshold: float,
) -> tuple[list[list[int]], list[list[float]], list[list[str]], list[dict[str, Any]], list[ScoreTask], list[int]]:
    output_records = []
    label_vectors = []
    raw_score_vectors = []
    status_vectors = []
    response_counts = []
    tasks = []

    for record_index, record in enumerate(records):
        reward_model = record.get(reward_model_key) or {}
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is None:
            raise KeyError(f"Missing ground truth in reward model field: {record.get(reward_model_key)}")

        data_source = record.get(data_source_key)
        if data_source is None:
            raise KeyError(f"Missing data source field: {data_source_key}")

        responses = ensure_list(record.get(response_key))
        output_records.append(dict(record))
        label_vectors.append([0] * len(responses))
        raw_score_vectors.append([0.0] * len(responses))
        status_vectors.append(["pending"] * len(responses))
        response_counts.append(len(responses))
        extra_info = record.get(extra_info_key)

        for response_index, response in enumerate(responses):
            tasks.append(
                ScoreTask(
                    record_index=record_index,
                    response_index=response_index,
                    data_source=str(data_source),
                    model_output=str(response),
                    ground_truth=str(ground_truth),
                    extra_info=extra_info,
                    timeout_seconds=timeout_seconds,
                    label_threshold=label_threshold,
                )
            )

    return label_vectors, raw_score_vectors, status_vectors, output_records, tasks, response_counts


def score_one_response(task: ScoreTask) -> ScoreResult:
    try:
        result = default_compute_score(
            data_source=task.data_source,
            solution_str=task.model_output,
            ground_truth=task.ground_truth,
            extra_info=task.extra_info,
        )
        raw_score = score_to_float(result)
        label = int(raw_score >= task.label_threshold)
        return ScoreResult(task.record_index, task.response_index, label, raw_score, "ok")
    except Exception:
        return ScoreResult(task.record_index, task.response_index, 0, 0.0, "error")



def summarize_question_distribution(score_vectors: list[list[int]]) -> None:
    total_questions = len(score_vectors)
    if total_questions == 0:
        print("No questions found.")
        return

    response_count_set = {len(scores) for scores in score_vectors}
    if len(response_count_set) == 1:
        response_count = next(iter(response_count_set))
        histogram = Counter(sum(scores) for scores in score_vectors)
        print(f"Question-level correctness distribution ({response_count} responses per question):")
        for correct_answers in range(response_count, -1, -1):
            count = histogram.get(correct_answers, 0)
            ratio = count / total_questions
            print(f"  {correct_answers}/{response_count} correct: {count} ({ratio:.2%})")
        return

    print("Question-level correctness distribution grouped by response count:")
    grouped_vectors: dict[int, list[list[int]]] = {}
    for scores in score_vectors:
        grouped_vectors.setdefault(len(scores), []).append(scores)

    for response_count in sorted(grouped_vectors.keys(), reverse=True):
        group_scores = grouped_vectors[response_count]
        group_total = len(group_scores)
        histogram = Counter(sum(scores) for scores in group_scores)
        print(f"  Questions with {response_count} responses: {group_total}")
        for correct_answers in range(response_count, -1, -1):
            count = histogram.get(correct_answers, 0)
            ratio = count / group_total
            print(f"    {correct_answers}/{response_count} correct: {count} ({ratio:.2%})")


def persist_scores(
    records: list[dict[str, Any]],
    label_vectors: list[list[int]],
    raw_score_vectors: list[list[float]],
    status_vectors: list[list[str]],
    output_path: Path,
    score_key: str,
    raw_score_key: str,
    status_key: str,
) -> None:
    for record, labels, raw_scores, statuses in zip(
        records, label_vectors, raw_score_vectors, status_vectors, strict=True
    ):
        record[score_key] = labels
        if raw_score_key:
            record[raw_score_key] = raw_scores
        if status_key:
            record[status_key] = statuses
    save_records_as_parquet(records, output_path)


def score_tasks(tasks: Iterable[ScoreTask], num_workers: int, chunksize: int) -> tuple[list[ScoreResult], Counter[str]]:
    status_counter: Counter[str] = Counter()
    results: list[ScoreResult] = []

    if num_workers <= 1:
        for task in tasks:
            result = score_one_response(task)
            status_counter[result.status] += 1
            results.append(result)
        return results, status_counter

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(score_one_response, tasks, chunksize=chunksize):
            status_counter[result.status] += 1
            results.append(result)

    return results, status_counter


def main() -> None:
    args = parse_args()
    records = load_records(args.input_path)
    label_vectors, raw_score_vectors, status_vectors, output_records, tasks, response_counts = build_score_tasks(
        records=records,
        response_key=args.response_key,
        reward_model_key=args.reward_model_key,
        data_source_key=args.data_source_key,
        extra_info_key=args.extra_info_key,
        timeout_seconds=args.timeout_seconds,
        label_threshold=args.label_threshold,
    )

    results, status_counter = score_tasks(tasks, num_workers=args.num_workers, chunksize=args.chunksize)
    total_correct = 0
    for result in results:
        label_vectors[result.record_index][result.response_index] = result.label
        raw_score_vectors[result.record_index][result.response_index] = result.raw_score
        status_vectors[result.record_index][result.response_index] = result.status
        total_correct += result.label

    persist_scores(
        records=output_records,
        label_vectors=label_vectors,
        raw_score_vectors=raw_score_vectors,
        status_vectors=status_vectors,
        output_path=args.output_path,
        score_key=args.score_key,
        raw_score_key=args.raw_score_key,
        status_key=args.status_key,
    )

    total_responses = sum(response_counts)
    response_accuracy = total_correct / max(total_responses, 1)
    print(f"Saved trainer-verifier scored rollouts to {args.output_path}")
    print(f"Total questions: {len(output_records)}")
    print(f"Total responses: {total_responses}")
    print(f"Correct responses: {total_correct}")
    print(f"Response accuracy: {response_accuracy:.4f}")
    print(f"Timed out responses: {status_counter.get('timeout', 0)}")
    print(f"Verifier errors treated as incorrect: {status_counter.get('error', 0)}")
    summarize_question_distribution(label_vectors)


if __name__ == "__main__":
    main()


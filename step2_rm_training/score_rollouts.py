from __future__ import annotations

"""Concurrent verifier scoring for rollout trajectories."""

import argparse
import os
import signal
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from common import load_records, save_records_as_parquet
from verifier_math_verify import ExprExtractionConfig, IMPORT_ERROR, LatexExtractionConfig, math_metric


@dataclass(frozen=True)
class ScoreTask:
    """A single verifier call for one response."""

    record_index: int
    response_index: int
    model_output: str
    ground_truth: str
    timeout_seconds: float
    label_threshold: float


@dataclass(frozen=True)
class ScoreResult:
    """Verifier result for one response."""

    record_index: int
    response_index: int
    label: int
    status: str


class VerifierTimeoutError(TimeoutError):
    """Raised when a single verifier call exceeds the allowed wall time."""


VERIFY_FUNC = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for attaching correctness labels to rollout samples."""
    parser = argparse.ArgumentParser(description="Score rollout responses with concurrent math_verify workers.")
    parser.add_argument("--input-path", type=str, required=True, help="Rollout dataset path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Scored dataset path.")
    parser.add_argument("--response-key", type=str, default="responses", help="Responses column name.")
    parser.add_argument("--reward-model-key", type=str, default="reward_model", help="Reward model metadata column name.")
    parser.add_argument("--score-key", type=str, default="score", help="Output field for per-response correctness labels.")
    parser.add_argument("--label-threshold", type=float, default=0.5, help="Threshold used to convert verifier output to correct/incorrect.")
    parser.add_argument("--timeout-seconds", type=float, default=1.0, help="Per-response verifier timeout in seconds.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Number of worker processes used for scoring.",
    )
    parser.add_argument("--chunksize", type=int, default=32, help="Chunksize passed to ProcessPoolExecutor.map.")
    return parser.parse_args()


def build_verify_func():
    """Create the math_verify callable once per worker process."""
    if IMPORT_ERROR is not None:
        raise ImportError("Please install math-verify before running step2 scoring.") from IMPORT_ERROR

    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def init_worker() -> None:
    """Initialize verifier state in each worker process."""
    global VERIFY_FUNC
    VERIFY_FUNC = build_verify_func()


def handle_timeout(_: int, __: Any) -> None:
    """Interrupt a verifier call after the configured timeout."""
    raise VerifierTimeoutError


def ensure_list_responses(responses: Any) -> list[Any]:
    """Normalize the response field into a list."""
    if responses is None:
        return []
    if isinstance(responses, list):
        return responses
    return [responses]


def build_score_tasks(
    records: list[dict[str, Any]],
    response_key: str,
    reward_model_key: str,
    timeout_seconds: float,
    label_threshold: float,
) -> tuple[list[list[int]], list[dict[str, Any]], list[ScoreTask], list[int]]:
    """Extract verifier tasks and allocate result storage."""
    task_records: list[dict[str, Any]] = []
    score_vectors: list[list[int]] = []
    response_counts: list[int] = []
    tasks: list[ScoreTask] = []

    for record_index, record in enumerate(records):
        reward_model = record.get(reward_model_key) or {}
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is None:
            raise KeyError(f"Missing ground truth in reward model field: {record.get(reward_model_key)}")

        responses = ensure_list_responses(record.get(response_key))
        task_records.append(dict(record))
        score_vectors.append([0] * len(responses))
        response_counts.append(len(responses))

        for response_index, response in enumerate(responses):
            tasks.append(
                ScoreTask(
                    record_index=record_index,
                    response_index=response_index,
                    model_output=str(response),
                    ground_truth=str(ground_truth),
                    timeout_seconds=timeout_seconds,
                    label_threshold=label_threshold,
                )
            )

    return score_vectors, task_records, tasks, response_counts


def score_one_response(task: ScoreTask) -> ScoreResult:
    """Score one response, forcing timeout cases to be marked incorrect."""
    global VERIFY_FUNC
    if VERIFY_FUNC is None:
        VERIFY_FUNC = build_verify_func()

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, task.timeout_seconds)

    try:
        ground_truth_boxed = f"\\boxed{{{task.ground_truth}}}"
        try:
            raw_score, _ = VERIFY_FUNC([ground_truth_boxed], [task.model_output])
        except VerifierTimeoutError:
            raise
        except Exception:
            return ScoreResult(task.record_index, task.response_index, 0, "error")
        label = int(float(raw_score) >= task.label_threshold)
        return ScoreResult(task.record_index, task.response_index, label, "ok")
    except VerifierTimeoutError:
        return ScoreResult(task.record_index, task.response_index, 0, "timeout")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def summarize_question_distribution(score_vectors: list[list[int]]) -> None:
    """Print question-level distribution by number of correct responses."""
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


def persist_scores(records: list[dict[str, Any]], score_vectors: list[list[int]], output_path: Path, score_key: str) -> None:
    """Attach score vectors to records and persist them."""
    for record, scores in zip(records, score_vectors):
        record[score_key] = scores
    save_records_as_parquet(records, output_path)


def score_tasks(tasks: Iterable[ScoreTask], num_workers: int, chunksize: int) -> tuple[list[ScoreResult], Counter[str]]:
    """Run verifier tasks concurrently and collect status counts."""
    status_counter: Counter[str] = Counter()
    results: list[ScoreResult] = []

    if num_workers <= 1:
        for task in tasks:
            result = score_one_response(task)
            status_counter[result.status] += 1
            results.append(result)
        return results, status_counter

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        for result in executor.map(score_one_response, tasks, chunksize=chunksize):
            status_counter[result.status] += 1
            results.append(result)

    return results, status_counter


def main() -> None:
    """Attach per-response correctness labels and print question-level statistics."""
    args = parse_args()
    records = load_records(args.input_path)
    score_vectors, output_records, tasks, response_counts = build_score_tasks(
        records=records,
        response_key=args.response_key,
        reward_model_key=args.reward_model_key,
        timeout_seconds=args.timeout_seconds,
        label_threshold=args.label_threshold,
    )

    results, status_counter = score_tasks(tasks, num_workers=args.num_workers, chunksize=args.chunksize)
    total_correct = 0
    for result in results:
        score_vectors[result.record_index][result.response_index] = result.label
        total_correct += result.label

    persist_scores(output_records, score_vectors, args.output_path, args.score_key)

    total_responses = sum(response_counts)
    response_accuracy = total_correct / max(total_responses, 1)
    print(f"Saved scored rollouts to {args.output_path}")
    print(f"Total questions: {len(output_records)}")
    print(f"Total responses: {total_responses}")
    print(f"Correct responses: {total_correct}")
    print(f"Response accuracy: {response_accuracy:.4f}")
    print(f"Timed out responses: {status_counter.get('timeout', 0)}")
    print(f"Verifier errors treated as incorrect: {status_counter.get('error', 0)}")
    summarize_question_distribution(score_vectors)


if __name__ == "__main__":
    main()

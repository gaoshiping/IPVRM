from __future__ import annotations

"""Math verifier wrapper that produces terminal correctness labels for RM training."""

import contextlib

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as import_error:  # pragma: no cover - runtime dependency
    math_metric = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    IMPORT_ERROR = import_error
else:
    IMPORT_ERROR = None


def compute_score(model_output: str, ground_truth: str) -> float:
    """Return the outcome label used for verifiable reasoning supervision.

    In the paper's Stage-A pipeline, RM training relies on trajectory-level
    correctness labels rather than explicit step annotations; this verifier
    supplies that terminal signal.
    """
    if IMPORT_ERROR is not None:
        raise ImportError("Please install math-verify before running step2 scoring.") from IMPORT_ERROR

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    with contextlib.suppress(Exception):
        score, _ = verify_func([ground_truth_boxed], [model_output])
    return float(score)
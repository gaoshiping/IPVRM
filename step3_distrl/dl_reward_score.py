from __future__ import annotations

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

VERIFY_FUNC = None


def _get_verify_func():
    global VERIFY_FUNC
    if VERIFY_FUNC is None:
        VERIFY_FUNC = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return VERIFY_FUNC


def step3_prime_verify_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    verify_func = _get_verify_func()
    ground_truth_boxed = f"\\boxed{{{ground_truth}}}"

    try:
        raw_score, _ = verify_func([ground_truth_boxed], [solution_str])
    except Exception:
        return 0.0

    return float(int(float(raw_score) >= 0.5))

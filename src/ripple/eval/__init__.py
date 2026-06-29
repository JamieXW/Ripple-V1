"""Evaluation (M2, build early): metrics for semantic queries (precision@k, recall@k,
nDCG@k) and impact queries (predicted blast radius vs. files commits actually changed),
always reported before/after on a held-out split."""

from ripple.eval.impact_eval import (
    EvalExample,
    build_examples,
    evaluate,
    run_impact_eval,
)
from ripple.eval.metrics import ImpactEvalReport, precision_recall

__all__ = [
    "EvalExample",
    "ImpactEvalReport",
    "build_examples",
    "evaluate",
    "precision_recall",
    "run_impact_eval",
]

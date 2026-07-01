"""Evaluation (M2 impact, M5a semantic): impact queries graded against files commits
actually changed; semantic queries graded on held-out docstring->function pairs
(recall@k, MRR, nDCG@k). Always before/after on a held-out split."""

from ripple.eval.impact_eval import (
    EvalExample,
    build_examples,
    evaluate,
    run_impact_eval,
)
from ripple.eval.metrics import (
    ImpactEvalReport,
    SearchEvalReport,
    ndcg_at_k,
    precision_at_k,
    precision_recall,
    recall_at_k,
    reciprocal_rank,
)
from ripple.eval.semantic_eval import build_eval_corpus, evaluate_search

__all__ = [
    "EvalExample",
    "ImpactEvalReport",
    "SearchEvalReport",
    "build_eval_corpus",
    "build_examples",
    "evaluate",
    "evaluate_search",
    "ndcg_at_k",
    "precision_at_k",
    "precision_recall",
    "recall_at_k",
    "reciprocal_rank",
    "run_impact_eval",
]

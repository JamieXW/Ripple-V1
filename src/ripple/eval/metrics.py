"""Evaluation metrics and the aggregate eval reports.

Impact eval uses set-based precision/recall over file paths: *precision* is how much of
what we predicted was real; *recall* is how much of what really changed we caught.
Precision is undefined when nothing was predicted (``None``), so it's averaged only over
examples that produced a prediction, while recall is averaged over every example.

Search eval uses standard ranking metrics over an ordered result list: recall@k (did the
right answer appear in the top k), MRR (how high, on average, via reciprocal rank), and
nDCG@k (ranking quality, rewarding hits nearer the top).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Of the top-``k`` returned, the fraction that are relevant."""
    if k <= 0:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Of all relevant items, the fraction appearing in the top-``k``."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit, or 0.0 if none — the summand of MRR."""
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG: hits near the top are worth more (1/log2(rank+1))."""
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def precision_recall(predicted: set[str], truth: set[str]) -> tuple[float | None, float]:
    """Return ``(precision, recall)`` for one example. Precision is ``None`` if nothing
    was predicted; recall is ``0.0`` if there's nothing to find."""
    hits = len(predicted & truth)
    precision = hits / len(predicted) if predicted else None
    recall = hits / len(truth) if truth else 0.0
    return precision, recall


@dataclass
class SearchEvalReport:
    """Aggregate results of the semantic-search eval on held-out docstring queries."""

    n_corpus: int  # chunks in the searchable corpus (docstrings stripped)
    n_queries: int  # held-out test queries
    n_train_pairs: int  # train-split pairs (reserved for M5b fine-tuning)
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


@dataclass
class ImpactEvalReport:
    """Aggregate results of grading impact predictions against git history."""

    n_commits: int
    n_examples: int  # (seed, ground-truth) pairs evaluated
    n_with_prediction: int  # examples where Ripple predicted >= 1 file
    mean_precision: float  # macro-avg over examples that predicted something
    mean_recall: float  # macro-avg over all examples
    mean_recall_when_predicted: float  # macro-avg recall over examples that predicted something
    micro_precision: float  # pooled hits / pooled predicted
    micro_recall: float  # pooled hits / pooled truth

    @property
    def coverage(self) -> float:
        """Fraction of examples for which Ripple predicted any blast radius at all.

        Overall recall ≈ coverage × recall-when-predicted, so low coverage (a static
        resolution limit) caps recall regardless of how good the predictions are.
        """
        return self.n_with_prediction / self.n_examples if self.n_examples else 0.0

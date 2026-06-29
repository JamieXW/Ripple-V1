"""Evaluation metrics and the aggregate impact-eval report.

Set-based precision/recall over file paths: *precision* is how much of what we
predicted was real; *recall* is how much of what really changed we caught. Precision
is undefined when nothing was predicted (``None``), so it's averaged only over examples
that produced a prediction, while recall is averaged over every example.
"""

from __future__ import annotations

from dataclasses import dataclass


def precision_recall(predicted: set[str], truth: set[str]) -> tuple[float | None, float]:
    """Return ``(precision, recall)`` for one example. Precision is ``None`` if nothing
    was predicted; recall is ``0.0`` if there's nothing to find."""
    hits = len(predicted & truth)
    precision = hits / len(predicted) if predicted else None
    recall = hits / len(truth) if truth else 0.0
    return precision, recall


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

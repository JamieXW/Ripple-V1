"""Mining (M2 + M5): generate training/eval data with no manual labeling — from
git history (commit -> changed files/functions) and docstrings (query -> code pairs);
graph-based hard negatives arrive with reranker training (M5b)."""

from ripple.mining.git_miner import CommitChange, mine_commits
from ripple.mining.negatives import TrainingExample, mine_training_examples
from ripple.mining.pairs import (
    QueryCodePair,
    mine_docstring_pairs,
    split_for,
    strip_docstring,
    stripped_chunks,
)

__all__ = [
    "CommitChange",
    "QueryCodePair",
    "TrainingExample",
    "mine_commits",
    "mine_docstring_pairs",
    "mine_training_examples",
    "split_for",
    "strip_docstring",
    "stripped_chunks",
]

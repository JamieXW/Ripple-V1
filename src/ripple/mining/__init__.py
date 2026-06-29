"""Mining (M2 + M5): generate training/eval data with no manual labeling — from
git history (commit -> changed files/functions), and later docstrings and call-graph
edges (+ hard negatives)."""

from ripple.mining.git_miner import CommitChange, mine_commits

__all__ = ["CommitChange", "mine_commits"]

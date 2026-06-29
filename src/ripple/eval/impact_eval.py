"""Grade Ripple's impact predictions against what git commits actually changed.

For each focused commit we turn every changed function that still exists at HEAD into
a *seed*: Ripple predicts the seed's blast radius (the files it thinks would be
affected), and we score that against the *other* files the commit actually touched.
This is the project's headline credibility check (CLAUDE.md §5.2 / §7).

Caveats, documented rather than hidden:
- Co-change is a noisy proxy for causal impact (unrelated edits share a commit).
- The graph is built at HEAD, so older commits referencing moved/removed code drift;
  we evaluate a recent window and only seeds that still exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from ripple.eval.metrics import ImpactEvalReport, precision_recall
from ripple.graph.builder import CodeGraph
from ripple.indexing.indexer import index_repo
from ripple.mining.git_miner import CommitChange, mine_commits


@dataclass(frozen=True)
class EvalExample:
    """One graded unit: changing ``seed`` should affect ``truth_files``."""

    commit: str
    seed: str  # qualified name of the changed function
    truth_files: frozenset[str]  # other .py files the commit touched


def build_examples(changes: list[CommitChange], graph: CodeGraph) -> list[EvalExample]:
    """Turn commit changes into gradable examples (seeds that exist in the graph)."""
    examples: list[EvalExample] = []
    for change in changes:
        for seed in change.changed_functions:
            node = graph.nodes.get(seed)
            if node is None:
                continue  # function no longer exists at HEAD (drift)
            truth = frozenset(f for f in change.changed_files if f != node.file_path)
            if truth:  # need at least one co-changed file to grade against
                examples.append(EvalExample(change.sha, seed, truth))
    return examples


def evaluate(graph: CodeGraph, examples: list[EvalExample]) -> ImpactEvalReport:
    """Score impact predictions for every example and aggregate."""
    precisions: list[float] = []
    recalls: list[float] = []
    recalls_when_predicted: list[float] = []
    hits_sum = predicted_sum = truth_sum = 0

    for example in examples:
        seed_file = graph.nodes[example.seed].file_path
        predicted = {a.node.file_path for a in graph.impact(example.seed).affected} - {seed_file}
        truth = set(example.truth_files)

        precision, recall = precision_recall(predicted, truth)
        recalls.append(recall)
        if precision is not None:
            precisions.append(precision)
            recalls_when_predicted.append(recall)

        hits_sum += len(predicted & truth)
        predicted_sum += len(predicted)
        truth_sum += len(truth)

    return ImpactEvalReport(
        n_commits=len({e.commit for e in examples}),
        n_examples=len(examples),
        n_with_prediction=len(precisions),
        mean_precision=fmean(precisions) if precisions else 0.0,
        mean_recall=fmean(recalls) if recalls else 0.0,
        mean_recall_when_predicted=fmean(recalls_when_predicted) if recalls_when_predicted else 0.0,
        micro_precision=hits_sum / predicted_sum if predicted_sum else 0.0,
        micro_recall=hits_sum / truth_sum if truth_sum else 0.0,
    )


def run_impact_eval(
    repo_path: Path,
    max_count: int = 300,
    min_files: int = 2,
    max_files: int = 10,
) -> ImpactEvalReport:
    """Index ``repo_path`` at HEAD, mine its history, and grade impact predictions."""
    graph = index_repo(repo_path)
    changes = mine_commits(repo_path, max_count=max_count, min_files=min_files, max_files=max_files)
    return evaluate(graph, build_examples(changes, graph))

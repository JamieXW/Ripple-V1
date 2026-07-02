"""Tests for hard-negative mining and the rerank path (fake models — no downloads)."""

from __future__ import annotations

from pathlib import Path

from ripple.embeddings.vector_index import VectorIndex
from ripple.eval import evaluate_search
from ripple.mining import mine_docstring_pairs, mine_training_examples, stripped_chunks
from ripple.parsing import parse_repo
from tests.test_embeddings import FakeEncoder


class PerfectReranker:
    """Scores by exact containment of the query's distinctive token — an oracle."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        token = next((w for w in query.split() if w.isdigit()), None)
        return [1.0 if token and f"+ {token}" in text else 0.0 for text in texts]


def _write_repo(root: Path, n: int = 30) -> None:
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    lines = []
    for i in range(n):
        lines.append(
            f'def func_{i}(x):\n    """Does distinctive thing number {i} to x."""\n'
            f"    return x + {i}\n\n"
        )
    (pkg / "m.py").write_text("".join(lines), encoding="utf-8")


def test_mine_training_examples_shapes_and_labels(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    modules = parse_repo(tmp_path)
    encoder = FakeEncoder()
    pairs = [p for p in mine_docstring_pairs(modules, tmp_path) if p.split == "train"]
    chunks = stripped_chunks(modules, tmp_path)
    corpus = VectorIndex(
        nodes=[n for n, _ in chunks],
        matrix=encoder.encode([t for _, t in chunks]),
        model_name=encoder.model_name,
    )
    query_matrix = encoder.encode([p.query for p in pairs])
    examples = mine_training_examples(pairs, corpus, [t for _, t in chunks], query_matrix)

    positives = [e for e in examples if e.label == 1.0]
    negatives = [e for e in examples if e.label == 0.0]
    assert len(positives) == len(pairs)
    assert len(negatives) > 0
    # A negative must never be the positive's own code.
    pos_code = {(e.query, e.code) for e in positives}
    assert all((e.query, e.code) not in pos_code for e in negatives)


def test_rerank_path_runs_and_oracle_wins(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    modules = parse_repo(tmp_path)
    baseline = evaluate_search(modules, tmp_path, FakeEncoder())
    reranked = evaluate_search(
        modules, tmp_path, FakeEncoder(), reranker=PerfectReranker(), retrieve_k=30
    )
    assert reranked.n_queries == baseline.n_queries
    # An oracle reranker over a wide candidate set can't do worse than the baseline.
    assert reranked.recall_at_1 >= baseline.recall_at_1

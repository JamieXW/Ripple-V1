"""Tests for pair mining (leakage safety), ranking metrics, and the search eval."""

from __future__ import annotations

from pathlib import Path

from ripple.eval import evaluate_search, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from ripple.mining import mine_docstring_pairs, split_for, strip_docstring
from ripple.parsing import parse_repo
from tests.test_embeddings import FakeEncoder

# --- metrics -------------------------------------------------------------------


def test_recall_and_precision_at_k() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"c"}, 2) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0
    assert precision_at_k(ranked, {"a", "c"}, 4) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "y", "hit"], {"hit"}) == 1 / 3
    assert reciprocal_rank(["x", "y"], {"hit"}) == 0.0


def test_ndcg_rewards_earlier_hits() -> None:
    early = ndcg_at_k(["hit", "x", "y"], {"hit"}, 10)
    late = ndcg_at_k(["x", "y", "hit"], {"hit"}, 10)
    assert early == 1.0
    assert 0.0 < late < early


# --- pair mining ----------------------------------------------------------------


def test_strip_docstring_removes_only_docstring() -> None:
    code = strip_docstring('def f():\n    """Docs here."""\n    return 1\n')
    assert code is not None
    assert "Docs here" not in code
    assert "return 1" in code


def test_strip_docstring_rejects_docstring_only_body() -> None:
    assert strip_docstring('def f():\n    """Only docs."""\n') is None


def test_split_is_deterministic() -> None:
    assert split_for("pkg.mod.func") == split_for("pkg.mod.func")


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


def test_mined_pairs_are_leakage_free(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    pairs = mine_docstring_pairs(parse_repo(tmp_path), tmp_path)
    assert len(pairs) > 0
    for pair in pairs:
        assert pair.query not in pair.code  # the docstring never appears in the code side
    assert {p.split for p in pairs} == {"train", "test"}  # both splits populated


# --- end-to-end eval with the fake encoder ---------------------------------------


def test_evaluate_search_returns_metrics(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    report = evaluate_search(parse_repo(tmp_path), tmp_path, FakeEncoder())
    assert report.n_queries > 0
    assert report.n_train_pairs > 0
    assert report.n_corpus >= report.n_queries
    for value in (report.recall_at_10, report.mrr, report.ndcg_at_10):
        assert 0.0 <= value <= 1.0

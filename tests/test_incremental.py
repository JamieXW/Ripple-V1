"""Incremental reindexing tests: pure diff logic + build behavior with a fake encoder."""

from __future__ import annotations

from pathlib import Path

from ripple.indexing import build_incremental, compute_file_hashes, plan_incremental
from tests.test_embeddings import FakeEncoder


def test_plan_diff_classifies_changed_added_removed() -> None:
    stored = {"a.py": "h1", "b.py": "h2", "gone.py": "h3"}
    current = {"a.py": "h1", "b.py": "CHANGED", "new.py": "h4"}
    plan = plan_incremental(stored, current)
    assert plan.changed == {"b.py", "new.py"}
    assert plan.removed == {"gone.py"}
    assert plan.unchanged == {"a.py"}
    assert not plan.is_noop


def test_plan_noop_when_hashes_match() -> None:
    hashes = {"a.py": "h1"}
    assert plan_incremental(hashes, dict(hashes)).is_noop


def _write_repo(root: Path) -> None:
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("def fa():\n    return fb()\n", encoding="utf-8")
    (pkg / "b.py").write_text("def fb():\n    return 1\n", encoding="utf-8")


def test_hashes_stable_and_change_sensitive(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    first = compute_file_hashes(tmp_path)
    assert first == compute_file_hashes(tmp_path)  # deterministic
    (tmp_path / "pkg" / "b.py").write_text("def fb():\n    return 2\n", encoding="utf-8")
    second = compute_file_hashes(tmp_path)
    assert first["pkg/b.py"] != second["pkg/b.py"]
    assert first["pkg/a.py"] == second["pkg/a.py"]


def test_incremental_build_embeds_only_changed_files(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    encoder = FakeEncoder()

    # First run: nothing stored -> everything is "changed" (full build).
    _, vectors, plan, hashes, report = build_incremental(tmp_path, {}, encoder)
    assert report.mode == "full"
    assert report.chunks_embedded == 2  # fa + fb

    # Touch one file: only its chunks re-embed; graph is still complete.
    (tmp_path / "pkg" / "b.py").write_text("def fb():\n    return 2\n", encoding="utf-8")
    graph, vectors, plan, _, report = build_incremental(tmp_path, hashes, encoder)
    assert report.mode == "incremental"
    assert plan.changed == {"pkg/b.py"}
    assert [n.qualified_name for n in vectors.nodes] == ["pkg.b.fb"]
    assert report.chunks_embedded == 1
    assert "pkg.a.fa" in graph.nodes  # unchanged file still fully in the graph
    assert ("pkg.a.fa", "pkg.b.fb") in graph.graph.edges()

    # No changes at all -> noop.
    current = compute_file_hashes(tmp_path)
    _, _, plan2, _, report2 = build_incremental(tmp_path, current, encoder)
    assert report2.mode == "noop"
    assert report2.chunks_embedded == 0

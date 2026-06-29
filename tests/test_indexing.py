"""Indexing tests: end-to-end index of a real on-disk repo + persistence roundtrip."""

from __future__ import annotations

from pathlib import Path

from ripple.indexing import graph_path, index_repo, load_graph, save_graph


def _write_sample_repo(root: Path) -> None:
    pkg = root / "sample_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text(
        "from sample_pkg.b import helper\n\ndef top():\n    return helper()\n",
        encoding="utf-8",
    )
    (pkg / "b.py").write_text(
        "def helper():\n    return leaf()\n\ndef leaf():\n    return 1\n",
        encoding="utf-8",
    )


def test_index_repo_builds_graph(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    graph = index_repo(tmp_path)
    assert "sample_pkg.b.leaf" in graph.nodes
    assert graph.graph.number_of_edges() >= 2


def test_save_load_roundtrip_preserves_impact(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    save_graph(index_repo(tmp_path), base=tmp_path)
    assert graph_path(tmp_path).exists()

    graph = load_graph(base=tmp_path)
    result = graph.impact(graph.resolve_symbol("leaf")[0])
    names = {a.node.qualified_name for a in result.affected}
    assert names == {"sample_pkg.b.helper", "sample_pkg.a.top"}


def test_load_without_index_raises(tmp_path: Path) -> None:
    try:
        load_graph(base=tmp_path)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")

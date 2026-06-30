"""Embedding tests: chunking, vector search math, and index build.

These use a deterministic fake encoder so no model is downloaded or run — the real
model is exercised manually in the Flask smoke run, not in unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ripple.embeddings import VectorIndex, build_vector_index, iter_chunks
from ripple.parsing import parse_repo
from ripple.parsing.models import CodeNode


class FakeEncoder:
    """Maps text -> a normalized vector via a tiny bag-of-keywords scheme."""

    model_name = "fake-test-encoder"
    vocab = ("json", "auth", "parse", "render")

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        rows = []
        for text in texts:
            counts = np.array([float(text.lower().count(w)) for w in self.vocab], dtype=np.float32)
            norm = np.linalg.norm(counts)
            rows.append(counts / norm if norm else counts)
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), len(self.vocab))


def _node(qn: str, kind: str = "function") -> CodeNode:
    return CodeNode(qn, kind, "m.py", 1, 2, None)


def test_search_ranks_by_cosine_similarity() -> None:
    nodes = [_node("a"), _node("b")]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = VectorIndex(nodes=nodes, matrix=matrix, model_name="x")
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [n.qualified_name for n, _ in results] == ["a", "b"]
    assert results[0][1] == 1.0  # exact match


def test_search_on_empty_index_returns_nothing() -> None:
    index = VectorIndex(nodes=[], matrix=np.zeros((0, 0), dtype=np.float32), model_name="x")
    assert index.search(np.array([1.0], dtype=np.float32)) == []


def test_iter_chunks_extracts_function_source(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "m.py").write_text("def parse_json():\n    return 1\n", encoding="utf-8")
    chunks = iter_chunks(parse_repo(tmp_path), tmp_path)
    by_name = {node.qualified_name: text for node, text in chunks}
    assert "def parse_json():" in by_name["pkg.m.parse_json"]


def test_build_vector_index_then_search_finds_relevant_chunk(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text(
        "def load_json(data):\n    # json json json\n    return data\n", encoding="utf-8"
    )
    (pkg / "b.py").write_text(
        "def check_auth(user):\n    # auth auth auth\n    return user\n", encoding="utf-8"
    )
    encoder = FakeEncoder()
    index = build_vector_index(parse_repo(tmp_path), tmp_path, encoder)
    assert len(index.nodes) == 2

    query = encoder.encode(["json"])[0]
    top_node, _ = index.search(query, k=1)[0]
    assert top_node.qualified_name == "pkg.a.load_json"

"""AST-aware chunking + the in-memory vector index (M3).

Chunks reuse M1's parse: one chunk per function/class, its exact source slice. The
:class:`VectorIndex` stores those chunks' normalized embeddings as a matrix and answers
nearest-neighbor queries with a single dot product (= cosine, since rows are normalized).
Brute-force is exact and instant at this scale; ANN indexing arrives with pgvector (M4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ripple.parsing.models import CodeNode, ParsedModule

#: Node kinds we embed and make searchable.
CHUNK_KINDS = frozenset({"function", "class"})


class TextEncoder(Protocol):
    """Anything that can turn texts into normalized row vectors (e.g. ``Embedder``)."""

    model_name: str

    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


def iter_chunks(modules: list[ParsedModule], repo_root: Path) -> list[tuple[CodeNode, str]]:
    """Pair each function/class node with its source slice (signature + body)."""
    chunks: list[tuple[CodeNode, str]] = []
    line_cache: dict[str, list[str]] = {}
    for module in modules:
        if module.file_path not in line_cache:
            try:
                text = (repo_root / module.file_path).read_text(encoding="utf-8")
                line_cache[module.file_path] = text.splitlines()
            except (OSError, UnicodeDecodeError):
                line_cache[module.file_path] = []
        lines = line_cache[module.file_path]
        for node in module.nodes:
            if node.kind not in CHUNK_KINDS:
                continue
            snippet = "\n".join(lines[node.start_line - 1 : node.end_line]).strip()
            if snippet:
                chunks.append((node, snippet))
    return chunks


@dataclass
class VectorIndex:
    """Function/class chunks and their normalized embedding matrix."""

    nodes: list[CodeNode]
    matrix: NDArray[np.float32]  # shape (N, d), L2-normalized rows
    model_name: str
    texts: list[str] | None = None  # chunk source text, parallel to nodes (for reranking)

    def search(self, query_vector: NDArray[np.float32], k: int = 5) -> list[tuple[CodeNode, float]]:
        """Return the top-``k`` (node, similarity) pairs by cosine similarity."""
        if self.matrix.shape[0] == 0:
            return []
        scores = self.matrix @ query_vector
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.nodes[int(i)], float(scores[int(i)])) for i in top]


def build_vector_index(
    modules: list[ParsedModule], repo_root: Path, encoder: TextEncoder
) -> VectorIndex:
    """Chunk ``modules`` and embed every chunk into a :class:`VectorIndex`."""
    chunks = iter_chunks(modules, repo_root)
    nodes = [node for node, _ in chunks]
    texts = [text for _, text in chunks]
    matrix = encoder.encode(texts)
    return VectorIndex(nodes=nodes, matrix=matrix, model_name=encoder.model_name, texts=texts)

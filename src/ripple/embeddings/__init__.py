"""Embeddings (M3): AST-aware chunking (by function/class), embed each chunk, and
nearest-neighbor search. Tier-0 off-the-shelf model first; pgvector ANN in M4."""

from ripple.embeddings.embedder import Embedder
from ripple.embeddings.vector_index import (
    CHUNK_KINDS,
    TextEncoder,
    VectorIndex,
    build_vector_index,
    iter_chunks,
)

__all__ = [
    "CHUNK_KINDS",
    "Embedder",
    "TextEncoder",
    "VectorIndex",
    "build_vector_index",
    "iter_chunks",
]

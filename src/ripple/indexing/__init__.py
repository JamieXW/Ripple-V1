"""Indexing (M1 full, M7 incremental): orchestrate parse -> graph + embed -> store.
Incremental reindexing re-processes only changed files + affected dependents,
detected via per-file content hashes (Merkle approach)."""

from ripple.indexing.indexer import (
    graph_path,
    index_repo,
    load_graph,
    load_vectors,
    save_graph,
    save_vectors,
    vectors_path,
)

__all__ = [
    "graph_path",
    "index_repo",
    "load_graph",
    "load_vectors",
    "save_graph",
    "save_vectors",
    "vectors_path",
]

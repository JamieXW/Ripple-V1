"""Indexing (M1 full, M7 incremental): orchestrate parse -> graph + embed -> store.
Incremental reindexing re-embeds only changed files (content hashes detect change);
parse + graph rebuild in full because they're cheap and staleness-proof."""

from ripple.indexing.incremental import (
    IncrementalPlan,
    IndexReport,
    build_incremental,
    compute_file_hashes,
    plan_incremental,
)
from ripple.indexing.indexer import (
    graph_path,
    index_repo,
    index_repository,
    load_graph,
    load_vectors,
    save_graph,
    save_vectors,
    vectors_path,
)

__all__ = [
    "IncrementalPlan",
    "IndexReport",
    "build_incremental",
    "compute_file_hashes",
    "graph_path",
    "index_repo",
    "index_repository",
    "load_graph",
    "load_vectors",
    "plan_incremental",
    "save_graph",
    "save_vectors",
    "vectors_path",
]

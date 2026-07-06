"""Indexing orchestration: parse a repo, build indexes, persist to Postgres.

``index_repository`` is the single entry point used by the CLI and the API: it runs
the incremental build (see incremental.py) and applies it to storage, timing every
stage. The pickle helpers below are the M1-era local format, kept for tests and
lightweight use.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

from ripple.db.repository import apply_incremental, load_file_hashes, write_index
from ripple.db.session import init_db, session_scope
from ripple.embeddings.vector_index import TextEncoder, VectorIndex
from ripple.graph.builder import CodeGraph, build_graph
from ripple.indexing.incremental import IndexReport, build_incremental
from ripple.parsing.parser import parse_repo

INDEX_DIRNAME = ".ripple"
GRAPH_FILENAME = "graph.pkl"
VECTORS_FILENAME = "vectors.pkl"


def index_repository(
    repo_root: Path, encoder: TextEncoder, full: bool = False
) -> tuple[CodeGraph, IndexReport]:
    """Index ``repo_root`` into Postgres — incrementally unless ``full`` is set."""
    init_db()
    with session_scope() as session:
        stored = {} if full else load_file_hashes(session)
        graph, vectors, plan, current_hashes, report = build_incremental(repo_root, stored, encoder)
        start = time.perf_counter()
        if not stored:
            report.mode = "full"
            write_index(session, graph, vectors, current_hashes)
        elif plan.is_noop:
            report.mode = "noop"
        else:
            apply_incremental(session, graph, vectors, plan.changed, plan.removed, current_hashes)
        report.timings_ms["db"] = round((time.perf_counter() - start) * 1000, 1)
        report.timings_ms["total"] = round(sum(report.timings_ms.values()), 1)
    return graph, report


def index_repo(repo_root: Path) -> CodeGraph:
    """Parse every Python file under ``repo_root`` and build the dependency graph."""
    modules = parse_repo(repo_root)
    return build_graph(modules, repo_root=str(repo_root))


def graph_path(base: Path | None = None) -> Path:
    """Location of the persisted graph, relative to ``base`` (default: cwd)."""
    return (base or Path.cwd()) / INDEX_DIRNAME / GRAPH_FILENAME


def save_graph(graph: CodeGraph, base: Path | None = None) -> Path:
    """Persist ``graph`` and return where it was written."""
    path = graph_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)
    return path


def load_graph(base: Path | None = None) -> CodeGraph:
    """Load the persisted graph. Raises ``FileNotFoundError`` if not indexed yet."""
    path = graph_path(base)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, CodeGraph):  # defensive: stale/foreign pickle
        raise TypeError(f"unexpected object in {path}: {type(graph)!r}")
    return graph


def vectors_path(base: Path | None = None) -> Path:
    """Location of the persisted vector index, relative to ``base`` (default: cwd)."""
    return (base or Path.cwd()) / INDEX_DIRNAME / VECTORS_FILENAME


def save_vectors(index: VectorIndex, base: Path | None = None) -> Path:
    """Persist ``index`` and return where it was written."""
    path = vectors_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(index, handle)
    return path


def load_vectors(base: Path | None = None) -> VectorIndex:
    """Load the persisted vector index. Raises ``FileNotFoundError`` if not built yet."""
    path = vectors_path(base)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        index = pickle.load(handle)
    if not isinstance(index, VectorIndex):  # defensive: stale/foreign pickle
        raise TypeError(f"unexpected object in {path}: {type(index)!r}")
    return index

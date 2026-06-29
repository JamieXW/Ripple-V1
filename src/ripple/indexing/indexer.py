"""Indexing orchestration: parse a repo into a graph and persist it.

For M1 the index is a single pickled :class:`CodeGraph` under ``.ripple/`` in the
current working directory — one active index at a time. M4 replaces this with a
Postgres-backed store; M7 makes updates incremental.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ripple.graph.builder import CodeGraph, build_graph
from ripple.parsing.parser import parse_repo

INDEX_DIRNAME = ".ripple"
GRAPH_FILENAME = "graph.pkl"


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

"""Incremental reindexing (M7): re-embed only what changed.

Embedding dominates index cost (~95% of wall time) and each chunk's vector depends
*only on its own text* — so vectors for unchanged files are provably still valid and
can be kept. The cheap stages (parse, name resolution, graph build) rerun in full,
which guarantees edge correctness with zero staleness reasoning: a changed file can
alter edges *from other files* (renames, deletions), so rebuilding the graph wholesale
is both simpler and safer, and it costs milliseconds.

Change detection is a per-file sha256 fingerprint stored in ``file_hashes`` — compare,
and you get changed / added / removed for free.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from ripple.embeddings.vector_index import TextEncoder, VectorIndex, build_vector_index
from ripple.graph.builder import CodeGraph, build_graph
from ripple.parsing.parser import iter_python_files, parse_repo


@dataclass(frozen=True)
class IncrementalPlan:
    """Which files need work, computed purely from two hash maps."""

    changed: frozenset[str]  # new or modified — re-embed these files' chunks
    removed: frozenset[str]  # gone from disk — delete their chunks + hashes
    unchanged: frozenset[str]

    @property
    def is_noop(self) -> bool:
        return not self.changed and not self.removed


@dataclass
class IndexReport:
    """What one indexing run did, and where the time went."""

    mode: str  # "full" | "incremental" | "noop"
    files_total: int
    files_changed: int
    files_removed: int
    chunks_embedded: int
    timings_ms: dict[str, float] = field(default_factory=dict)


def compute_file_hashes(repo_root: Path) -> dict[str, str]:
    """sha256 fingerprint per indexable file (repo-relative paths)."""
    hashes: dict[str, str] = {}
    for path in iter_python_files(repo_root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(repo_root))] = digest
    return hashes


def plan_incremental(stored: dict[str, str], current: dict[str, str]) -> IncrementalPlan:
    """Diff two hash maps into changed / removed / unchanged sets. Pure function."""
    changed = {path for path, digest in current.items() if stored.get(path) != digest}
    removed = set(stored) - set(current)
    unchanged = set(current) - changed
    return IncrementalPlan(
        changed=frozenset(changed), removed=frozenset(removed), unchanged=frozenset(unchanged)
    )


def build_incremental(
    repo_root: Path,
    stored_hashes: dict[str, str],
    encoder: TextEncoder,
) -> tuple[CodeGraph, VectorIndex, IncrementalPlan, dict[str, str], IndexReport]:
    """Parse + graph everything, embed only changed files' chunks.

    Returns (graph, changed_vectors, plan, current_hashes, report); the caller applies
    the result to storage (see repository.apply_incremental / the CLI orchestration).
    """
    timings: dict[str, float] = {}

    start = time.perf_counter()
    current_hashes = compute_file_hashes(repo_root)
    plan = plan_incremental(stored_hashes, current_hashes)
    timings["hash_diff"] = round((time.perf_counter() - start) * 1000, 1)

    start = time.perf_counter()
    modules = parse_repo(repo_root)
    timings["parse"] = round((time.perf_counter() - start) * 1000, 1)

    start = time.perf_counter()
    graph = build_graph(modules, repo_root=str(repo_root))
    timings["graph"] = round((time.perf_counter() - start) * 1000, 1)

    start = time.perf_counter()
    changed_modules = [m for m in modules if m.file_path in plan.changed]
    vectors = build_vector_index(changed_modules, repo_root, encoder)
    timings["embed"] = round((time.perf_counter() - start) * 1000, 1)

    mode = "noop" if plan.is_noop else ("full" if not stored_hashes else "incremental")
    report = IndexReport(
        mode=mode,
        files_total=len(current_hashes),
        files_changed=len(plan.changed),
        files_removed=len(plan.removed),
        chunks_embedded=len(vectors.nodes),
        timings_ms=timings,
    )
    return graph, vectors, plan, current_hashes, report

"""Read/write the index to Postgres.

Writing replaces the whole index (single active repo for M4). Reading reconstructs an
in-memory :class:`CodeGraph` for impact traversal (the edge set is tiny, so loading it
and reusing the proven NetworkX walk beats a recursive SQL CTE here), while search runs
natively in Postgres via pgvector's HNSW index.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ripple.db.models import ChunkRow, EdgeRow, FileHashRow, NodeRow
from ripple.embeddings.vector_index import VectorIndex
from ripple.graph.builder import CodeGraph
from ripple.graph.models import ResolutionStats
from ripple.parsing.models import CodeNode


def write_index(
    session: Session,
    graph: CodeGraph,
    vectors: VectorIndex,
    file_hashes: dict[str, str] | None = None,
) -> None:
    """Full replace: all stored data becomes this graph + vector index (+ hashes)."""
    session.execute(delete(ChunkRow))
    replace_graph(session, graph)
    _add_chunks(session, vectors)
    session.execute(delete(FileHashRow))
    if file_hashes:
        session.add_all(
            FileHashRow(file_path=path, content_hash=digest) for path, digest in file_hashes.items()
        )


def _add_chunks(session: Session, vectors: VectorIndex) -> None:
    session.add_all(
        ChunkRow(
            qualified_name=node.qualified_name,
            file_path=node.file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            model_name=vectors.model_name,
            content=vectors.texts[i] if vectors.texts else "",
            embedding=vectors.matrix[i],
        )
        for i, node in enumerate(vectors.nodes)
    )


def replace_graph(session: Session, graph: CodeGraph) -> None:
    """Swap the stored nodes/edges for this graph (cheap; always rebuilt in full)."""
    session.execute(delete(EdgeRow))
    session.execute(delete(NodeRow))
    session.add_all(
        NodeRow(
            qualified_name=node.qualified_name,
            kind=node.kind,
            file_path=node.file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            docstring=node.docstring,
        )
        for node in graph.nodes.values()
    )
    session.add_all(
        EdgeRow(src=src, dst=dst, kind=data["kind"])
        for src, dst, data in graph.graph.edges(data=True)
    )


def load_file_hashes(session: Session) -> dict[str, str]:
    """Stored per-file fingerprints (empty dict on first index)."""
    return {
        row.file_path: row.content_hash for row in session.execute(select(FileHashRow)).scalars()
    }


def apply_incremental(
    session: Session,
    graph: CodeGraph,
    changed_vectors: VectorIndex,
    changed_files: frozenset[str],
    removed_files: frozenset[str],
    current_hashes: dict[str, str],
) -> None:
    """Apply one incremental build: swap graph, touch only changed/removed chunks."""
    replace_graph(session, graph)
    stale = changed_files | removed_files
    if stale:
        session.execute(delete(ChunkRow).where(ChunkRow.file_path.in_(stale)))
    _add_chunks(session, changed_vectors)
    session.execute(delete(FileHashRow))
    session.add_all(
        FileHashRow(file_path=path, content_hash=digest) for path, digest in current_hashes.items()
    )


def load_graph_from_db(session: Session) -> CodeGraph:
    """Reconstruct a :class:`CodeGraph` (nodes + edges) from the database."""
    nodes: dict[str, CodeNode] = {
        row.qualified_name: CodeNode(
            qualified_name=row.qualified_name,
            kind=row.kind,
            file_path=row.file_path,
            start_line=row.start_line,
            end_line=row.end_line,
            docstring=row.docstring,
        )
        for row in session.execute(select(NodeRow)).scalars()
    }
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row in session.execute(select(EdgeRow)).scalars():
        if row.src in nodes and row.dst in nodes:
            graph.add_edge(row.src, row.dst, kind=row.kind)
    return CodeGraph(graph, nodes, ResolutionStats())


def stored_model_name(session: Session) -> str | None:
    """The embedding model the current index was built with, or ``None`` if empty."""
    return session.execute(select(ChunkRow.model_name).limit(1)).scalar()


@dataclass(frozen=True)
class ChunkHit:
    """One retrieved chunk: where it is, how similar, and its source text."""

    node: CodeNode
    score: float
    content: str


def _hit(chunk: ChunkRow, score: float) -> ChunkHit:
    node = CodeNode(
        qualified_name=chunk.qualified_name,
        kind="function",
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line or chunk.start_line,
        docstring=None,
    )
    return ChunkHit(node=node, score=score, content=chunk.content)


def search_chunks(
    session: Session, query_vector: NDArray[np.float32], k: int = 5
) -> list[ChunkHit]:
    """Top-``k`` chunks by cosine similarity, using the pgvector HNSW index."""
    distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")
    rows = session.execute(select(ChunkRow, distance).order_by(distance).limit(k)).all()
    return [_hit(chunk, 1.0 - float(dist)) for chunk, dist in rows]


async def async_search_chunks(
    session: AsyncSession, query_vector: NDArray[np.float32], k: int = 5
) -> list[ChunkHit]:
    """Async variant of :func:`search_chunks` for the service layer."""
    distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")
    result = await session.execute(select(ChunkRow, distance).order_by(distance).limit(k))
    return [_hit(chunk, 1.0 - float(dist)) for chunk, dist in result.all()]


async def async_chunks_by_qnames(session: AsyncSession, qnames: list[str]) -> list[ChunkHit]:
    """Fetch specific chunks (e.g. graph-expansion candidates) by qualified name."""
    if not qnames:
        return []
    result = await session.execute(select(ChunkRow).where(ChunkRow.qualified_name.in_(qnames)))
    return [_hit(chunk, 0.0) for chunk in result.scalars()]

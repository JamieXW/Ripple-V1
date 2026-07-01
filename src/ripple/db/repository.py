"""Read/write the index to Postgres.

Writing replaces the whole index (single active repo for M4). Reading reconstructs an
in-memory :class:`CodeGraph` for impact traversal (the edge set is tiny, so loading it
and reusing the proven NetworkX walk beats a recursive SQL CTE here), while search runs
natively in Postgres via pgvector's HNSW index.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ripple.db.models import ChunkRow, EdgeRow, NodeRow
from ripple.embeddings.vector_index import VectorIndex
from ripple.graph.builder import CodeGraph
from ripple.graph.models import ResolutionStats
from ripple.parsing.models import CodeNode


def write_index(session: Session, graph: CodeGraph, vectors: VectorIndex) -> None:
    """Replace all stored data with this graph + vector index."""
    session.execute(delete(ChunkRow))
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
    session.add_all(
        ChunkRow(
            qualified_name=node.qualified_name,
            file_path=node.file_path,
            start_line=node.start_line,
            model_name=vectors.model_name,
            embedding=vectors.matrix[i],
        )
        for i, node in enumerate(vectors.nodes)
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


def search_chunks(
    session: Session, query_vector: NDArray[np.float32], k: int = 5
) -> list[tuple[CodeNode, float]]:
    """Top-``k`` chunks by cosine similarity, using the pgvector HNSW index."""
    distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")
    rows = session.execute(select(ChunkRow, distance).order_by(distance).limit(k)).all()
    results: list[tuple[CodeNode, float]] = []
    for chunk, dist in rows:
        node = CodeNode(
            qualified_name=chunk.qualified_name,
            kind="function",
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.start_line,
            docstring=None,
        )
        results.append((node, 1.0 - float(dist)))
    return results

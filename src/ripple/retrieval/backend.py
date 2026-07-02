"""Production wiring for the retrieval pipeline (shared by the API and MCP server).

Both entry points assemble the same components: the in-memory graph loaded from
Postgres, the embedder, the optional fine-tuned reranker, and pgvector-backed
retrieval. Kept here so there is exactly one definition of "the default pipeline".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.exc import OperationalError, ProgrammingError

from ripple.config import settings
from ripple.db.repository import (
    ChunkHit,
    async_chunks_by_qnames,
    async_search_chunks,
    load_graph_from_db,
)
from ripple.db.session import async_session_scope, session_scope
from ripple.embeddings.embedder import Embedder
from ripple.graph.builder import CodeGraph
from ripple.retrieval.pipeline import SearchPipeline
from ripple.retrieval.reranker import CrossEncoderReranker


async def pgvector_retrieve(query_vector: object, k: int) -> list[ChunkHit]:
    async with async_session_scope() as session:
        vector: NDArray[np.float32] = query_vector  # type: ignore[assignment]
        return await async_search_chunks(session, vector, k)


async def fetch_chunks(qnames: list[str]) -> list[ChunkHit]:
    async with async_session_scope() as session:
        return await async_chunks_by_qnames(session, qnames)


def load_graph_or_none() -> CodeGraph | None:
    """Load the graph from Postgres; ``None`` when the DB is down or empty."""
    try:
        with session_scope() as session:
            graph = load_graph_from_db(session)
        return graph if graph.nodes else None
    except (OperationalError, ProgrammingError):
        return None


def build_default_pipeline(
    graph: CodeGraph | None,
    embedder: Embedder | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> SearchPipeline:
    """The production pipeline: pgvector retrieval + settings-driven models."""
    if reranker is None and settings.reranker_model:
        reranker = CrossEncoderReranker(settings.reranker_model)
    return SearchPipeline(
        embedder=embedder or Embedder(),
        retrieve=pgvector_retrieve,
        fetch_chunks=fetch_chunks,
        graph=graph,
        reranker=reranker,
        retrieve_k=settings.retrieve_k,
        expand_cap=settings.expand_hops_cap,
    )

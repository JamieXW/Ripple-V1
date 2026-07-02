"""The hybrid retrieval pipeline (M6): semantic seed -> graph expand -> rerank.

This is the project's thesis assembled in one place. A query is embedded and the
vector store returns the top semantic *seeds*; the call graph then pulls in each
seed's direct neighbors (callers/callees) — code that's structurally related even if
it doesn't match the words; finally the cross-encoder reranks the combined pool.

Async shape: embedding and reranking are CPU-bound model inference, so they run in a
worker thread (``asyncio.to_thread``) — putting them on the event loop would block
every other request. Retrieval is I/O-bound database work and uses the async driver.
Each stage is wrapped in an OpenTelemetry span and its wall time is recorded, so every
response can say where its milliseconds went.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from opentelemetry import trace

from ripple.db.repository import ChunkHit
from ripple.graph.builder import CodeGraph
from ripple.retrieval.reranker import Reranker


class QueryEncoder(Protocol):
    """Anything that can embed a query string into a normalized vector."""

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


#: Fetches top-k nearest chunks for a query vector. Injected so tests need no database.
SeedRetriever = Callable[[object, int], Awaitable[list[ChunkHit]]]
#: Fetches chunks by qualified name (graph-expansion candidates).
ChunkFetcher = Callable[[list[str]], Awaitable[list[ChunkHit]]]

tracer = trace.get_tracer("ripple.retrieval")


@dataclass(frozen=True)
class SearchResult:
    """One ranked answer: the chunk plus how it entered the candidate pool."""

    hit: ChunkHit
    source: str  # "semantic" | "graph"


@contextmanager
def _stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    """OTel span + wall-time capture for one pipeline stage."""
    start = time.perf_counter()
    with tracer.start_as_current_span(f"ripple.{name}"):
        yield
    timings[name] = round((time.perf_counter() - start) * 1000, 2)


class SearchPipeline:
    """Retrieve-expand-rerank, with per-stage timings on every call."""

    def __init__(
        self,
        embedder: QueryEncoder,
        retrieve: SeedRetriever,
        fetch_chunks: ChunkFetcher,
        graph: CodeGraph | None = None,
        reranker: Reranker | None = None,
        retrieve_k: int = 50,
        expand_cap: int = 20,
    ) -> None:
        self.embedder = embedder
        self.retrieve = retrieve
        self.fetch_chunks = fetch_chunks
        self.graph = graph
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.expand_cap = expand_cap

    def _expansion_qnames(self, seeds: list[ChunkHit], k_seeds: int = 5) -> list[str]:
        """Direct graph neighbors (callers + callees) of the top seeds, capped."""
        if self.graph is None:
            return []
        have = {seed.node.qualified_name for seed in seeds}
        expanded: list[str] = []
        for seed in seeds[:k_seeds]:
            qname = seed.node.qualified_name
            if qname not in self.graph.graph:
                continue
            neighbors = list(self.graph.graph.predecessors(qname)) + list(
                self.graph.graph.successors(qname)
            )
            for neighbor in neighbors:
                if neighbor not in have:
                    have.add(neighbor)
                    expanded.append(neighbor)
                if len(expanded) >= self.expand_cap:
                    return expanded
        return expanded

    async def search(
        self,
        query: str,
        k: int = 5,
        rerank: bool = True,
        expand: bool = True,
    ) -> tuple[list[SearchResult], dict[str, float]]:
        timings: dict[str, float] = {}
        use_reranker = rerank and self.reranker is not None
        fetch_k = self.retrieve_k if use_reranker else max(k, 10)

        with _stage("embed", timings):
            query_vector = await asyncio.to_thread(self.embedder.embed_query, query)

        with _stage("retrieve", timings):
            seeds = await self.retrieve(query_vector, fetch_k)
        candidates: list[SearchResult] = [SearchResult(hit=s, source="semantic") for s in seeds]

        if expand:
            with _stage("expand", timings):
                extra = await self.fetch_chunks(self._expansion_qnames(seeds))
                candidates += [SearchResult(hit=hit, source="graph") for hit in extra]

        if use_reranker and self.reranker is not None:
            with _stage("rerank", timings):
                scores = await asyncio.to_thread(
                    self.reranker.score,
                    query,
                    [c.hit.content for c in candidates],
                )
            reranked = [
                SearchResult(
                    hit=ChunkHit(node=c.hit.node, score=score, content=c.hit.content),
                    source=c.source,
                )
                for c, score in zip(candidates, scores, strict=True)
            ]
            candidates = sorted(reranked, key=lambda r: -r.hit.score)
        else:
            candidates.sort(key=lambda r: -r.hit.score)

        timings["total"] = round(sum(timings.values()), 2)
        return candidates[:k], timings

"""Service-layer tests: hybrid pipeline (seed -> expand -> rerank) and the API.

All model/DB dependencies are injected fakes — no Postgres, Redis, or model downloads
required. The app's own lifespan runs (and degrades gracefully); we then override its
state with deterministic components.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from ripple.api.app import create_app
from ripple.db.repository import ChunkHit
from ripple.graph import build_graph
from ripple.graph.builder import CodeGraph
from ripple.parsing import parse_source
from ripple.parsing.models import CodeNode
from ripple.retrieval.pipeline import SearchPipeline


class FakeQueryEncoder:
    model_name = "fake-query-encoder"

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return np.array([1.0, 0.0], dtype=np.float32)


def _hit(qname: str, score: float) -> ChunkHit:
    short = qname.rsplit(".", 1)[-1]
    node = CodeNode(qname, "function", "pkg/m.py", 1, 2, None)
    return ChunkHit(node=node, score=score, content=f"def {short}(): pass")


async def retrieve_a_only(query_vector: object, k: int) -> list[ChunkHit]:
    return [_hit("pkg.m.a", 0.9)]


async def fetch_by_qname(qnames: list[str]) -> list[ChunkHit]:
    return [_hit(q, 0.0) for q in qnames]


def chain_graph() -> CodeGraph:
    # a -> b -> c: seeding with `a` should let expansion pull in its callee `b`.
    module = parse_source(
        "def a():\n    return b()\n\ndef b():\n    return c()\n\ndef c():\n    return 1\n",
        "pkg.m",
        "pkg/m.py",
    )
    return build_graph([module])


class PreferB:
    """Oracle reranker that loves function b."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 if "def b()" in t else 0.1 for t in texts]


# --- pipeline ---------------------------------------------------------------------


async def test_graph_expansion_adds_structural_neighbors() -> None:
    pipeline = SearchPipeline(
        FakeQueryEncoder(), retrieve_a_only, fetch_by_qname, graph=chain_graph()
    )
    results, timings = await pipeline.search("where is a", k=10, rerank=False)
    by_name = {r.hit.node.qualified_name: r.source for r in results}
    assert by_name["pkg.m.a"] == "semantic"
    assert by_name["pkg.m.b"] == "graph"  # entered via expansion, not similarity
    assert set(timings) >= {"embed", "retrieve", "expand", "total"}


async def test_reranker_reorders_candidates() -> None:
    pipeline = SearchPipeline(
        FakeQueryEncoder(),
        retrieve_a_only,
        fetch_by_qname,
        graph=chain_graph(),
        reranker=PreferB(),
    )
    results, timings = await pipeline.search("where is b", k=2, rerank=True)
    assert results[0].hit.node.qualified_name == "pkg.m.b"  # graph candidate won via rerank
    assert "rerank" in timings


async def test_expansion_disabled_stays_semantic_only() -> None:
    pipeline = SearchPipeline(
        FakeQueryEncoder(), retrieve_a_only, fetch_by_qname, graph=chain_graph()
    )
    results, _ = await pipeline.search("q", k=10, rerank=False, expand=False)
    assert {r.source for r in results} == {"semantic"}


# --- API --------------------------------------------------------------------------


def _client_with_fakes() -> TestClient:
    app = create_app()
    client = TestClient(app)
    client.__enter__()  # run lifespan (degrades gracefully without DB/Redis)
    graph = chain_graph()
    app.state.graph = graph
    app.state.cache.available = False
    app.state.pipeline = SearchPipeline(
        FakeQueryEncoder(), retrieve_a_only, fetch_by_qname, graph=graph
    )
    return client


def test_health_reports_index_cache_models() -> None:
    client = _client_with_fakes()
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]["nodes"] > 0
    assert "cache" in body and "models" in body


def test_search_returns_results_with_citations_and_timings() -> None:
    client = _client_with_fakes()
    body = client.get("/search", params={"q": "where is a", "rerank": False}).json()
    assert body["results"], body
    first = body["results"][0]
    assert {"symbol", "file", "line", "score", "source"} <= set(first)
    assert "timings_ms" in body["meta"]


def test_impact_returns_blast_radius() -> None:
    client = _client_with_fakes()
    body = client.get("/impact", params={"symbol": "c"}).json()
    names = {a["symbol"]: a["dependency"] for a in body["affected"]}
    assert names["pkg.m.b"] == "direct"
    assert names["pkg.m.a"] == "transitive"


def test_impact_unknown_symbol_is_404() -> None:
    client = _client_with_fakes()
    assert client.get("/impact", params={"symbol": "zzz"}).status_code == 404

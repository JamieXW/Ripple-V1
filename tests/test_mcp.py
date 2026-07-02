"""MCP server tests: tools against a fake engine + a real in-memory MCP round-trip."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ripple import mcp_server
from ripple.retrieval.pipeline import SearchPipeline
from tests.test_api import FakeQueryEncoder, chain_graph, fetch_by_qname, retrieve_a_only


@pytest.fixture(autouse=True)
def fake_engine() -> Iterator[None]:
    graph = chain_graph()
    mcp_server.engine.graph = graph
    mcp_server.engine.pipeline = SearchPipeline(
        FakeQueryEncoder(), retrieve_a_only, fetch_by_qname, graph=graph
    )
    mcp_server.engine.loaded = True
    yield
    mcp_server.engine.graph = None
    mcp_server.engine.pipeline = None
    mcp_server.engine.loaded = False


async def test_search_code_tool_returns_citations() -> None:
    body = await mcp_server.search_code("where is a", k=5, rerank=False)
    assert body["results"]
    first = body["results"][0]
    assert {"symbol", "file", "line", "score", "source"} <= set(first)
    assert "timings_ms" in body


def test_impact_tool_blast_radius_and_ambiguity() -> None:
    body = mcp_server.impact_of("c")
    deps = {a["symbol"]: a["dependency"] for a in body["affected"]}
    assert deps["pkg.m.b"] == "direct"
    assert deps["pkg.m.a"] == "transitive"

    missing = mcp_server.impact_of("zzz")
    assert "error" in missing


def test_index_status_tool() -> None:
    body = mcp_server.index_status()
    assert body["indexed"] is True
    assert body["nodes"] > 0


async def test_tools_reachable_over_real_mcp_session() -> None:
    """End-to-end over the actual protocol: list tools and call one in-memory."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert {"search_code", "impact_of", "index_status"} <= names

        result = await client.call_tool("impact_of", {"symbol": "c"})
        assert not result.isError

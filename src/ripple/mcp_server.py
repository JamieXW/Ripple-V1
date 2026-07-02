"""Ripple as an MCP server (M6.5): code intelligence as tools for AI agents.

Any MCP client (Claude Code, Claude Desktop, Cursor, ...) can connect over stdio and
call Ripple mid-task — most importantly ``impact_of`` *before editing a function*, the
question coding agents currently answer badly. Tool docstrings below are what the
agent reads to decide when to call each tool, so they're written for the model.

Runs in-process against the same components as the API (see retrieval/backend.py);
the FastAPI service does not need to be running. Start with ``ripple mcp`` or via a
client config like the committed ``.mcp.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import FastMCP

from ripple.config import settings
from ripple.graph.builder import CodeGraph
from ripple.retrieval.backend import build_default_pipeline, load_graph_or_none
from ripple.retrieval.pipeline import SearchPipeline

mcp = FastMCP(
    "ripple",
    instructions=(
        "Code-intelligence for the indexed Python repository. Call impact_of BEFORE "
        "modifying a function to learn what depends on it; call search_code to locate "
        "functionality by meaning. All results carry file:line citations."
    ),
)


@dataclass
class _Engine:
    """Lazily-assembled shared state (graph + pipeline), swappable in tests."""

    graph: CodeGraph | None = None
    pipeline: SearchPipeline | None = None
    loaded: bool = field(default=False)

    def ensure(self) -> None:
        if not self.loaded:
            self.graph = load_graph_or_none()
            self.pipeline = build_default_pipeline(self.graph)
            self.loaded = True


engine = _Engine()


@mcp.tool()
async def search_code(query: str, k: int = 5, rerank: bool = True) -> dict[str, Any]:
    """Find code in the indexed repository by meaning, not keywords.

    Use this to locate where functionality lives (e.g. "where are sessions loaded from
    cookies?"). Returns ranked matches with file path and line number citations; the
    `source` field says whether a match was found by semantic similarity or pulled in
    via the call graph.
    """
    engine.ensure()
    if engine.pipeline is None:
        return {"error": "no index loaded — run `ripple index <repo>` first"}
    results, timings = await engine.pipeline.search(query, k=k, rerank=rerank)
    return {
        "query": query,
        "results": [
            {
                "symbol": r.hit.node.qualified_name,
                "file": r.hit.node.file_path,
                "line": r.hit.node.start_line,
                "score": round(r.hit.score, 4),
                "source": r.source,
            }
            for r in results
        ],
        "timings_ms": timings,
    }


@mcp.tool()
def impact_of(symbol: str) -> dict[str, Any]:
    """What breaks if this function/class/method changes? Call BEFORE editing code.

    Returns the blast radius: every function that directly or transitively calls (or
    inherits from) the symbol, with file:line citations. Accepts a bare name (`save`),
    a partial (`Admin.save`), or a fully qualified name (`pkg.auth.Admin.save`). If the
    name is ambiguous, the response lists candidates to retry with. Note: static
    analysis under-reports dynamic dispatch — an empty result lowers but does not
    eliminate risk.
    """
    engine.ensure()
    if engine.graph is None:
        return {"error": "no index loaded — run `ripple index <repo>` first"}
    matches = engine.graph.resolve_symbol(symbol)
    if not matches:
        return {"error": f"symbol not found: {symbol!r}"}
    if len(matches) > 1:
        return {"ambiguous": True, "candidates": matches[:15]}
    result = engine.graph.impact(matches[0])
    return {
        "symbol": result.target.qualified_name,
        "location": f"{result.target.file_path}:{result.target.start_line}",
        "affected": [
            {
                "symbol": a.node.qualified_name,
                "file": a.node.file_path,
                "line": a.node.start_line,
                "dependency": "direct" if a.direct else "transitive",
            }
            for a in result.affected
        ],
        "counts": {"direct": result.direct_count, "total": len(result.affected)},
    }


@mcp.tool()
def index_status() -> dict[str, Any]:
    """Is a repository indexed, and how big is the graph? Call if other tools error."""
    engine.ensure()
    if engine.graph is None:
        return {"indexed": False, "hint": "run `ripple index <repo>` to build the index"}
    return {
        "indexed": True,
        "nodes": engine.graph.graph.number_of_nodes(),
        "edges": engine.graph.graph.number_of_edges(),
        "reranker": settings.reranker_model or None,
    }


def run() -> None:
    """Run the MCP server over stdio (blocking)."""
    mcp.run()

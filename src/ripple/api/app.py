"""Ripple's async service layer (M6).

Endpoints (all answers carry file:line citations):
- ``GET  /search``  — hybrid retrieval: semantic seed -> graph expand -> rerank.
- ``GET  /impact``  — blast radius via in-memory graph traversal.
- ``POST /index``   — kick off (re)indexing in the background.
- ``GET  /health``  — liveness, index status, cache hit rate, model names.

Degrades gracefully: with no database the app still boots (empty index, /health says
so); with no Redis, caching turns off. The graph lives in memory (loaded at startup,
swapped atomically after reindex) — it's small; the vectors stay in Postgres.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, ProgrammingError

from ripple.api.cache import ResponseCache
from ripple.api.tracing import install_middleware, setup_tracing
from ripple.config import settings
from ripple.embeddings.embedder import Embedder
from ripple.graph.builder import CodeGraph
from ripple.indexing import index_repository
from ripple.retrieval.backend import build_default_pipeline, load_graph_or_none
from ripple.retrieval.pipeline import SearchResult
from ripple.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_tracing()
    app.state.embedder = Embedder()
    app.state.reranker = (
        CrossEncoderReranker(settings.reranker_model) if settings.reranker_model else None
    )
    app.state.graph = await asyncio.to_thread(load_graph_or_none)
    app.state.cache = ResponseCache(settings.redis_url, settings.cache_ttl_seconds)
    await app.state.cache.connect()
    app.state.pipeline = build_default_pipeline(
        app.state.graph, embedder=app.state.embedder, reranker=app.state.reranker
    )
    app.state.index_status = {"state": "ready" if app.state.graph else "empty"}
    logger.info(
        "ripple api up — graph=%s, cache=%s, reranker=%s",
        "loaded" if app.state.graph else "empty",
        "on" if app.state.cache.available else "off",
        settings.reranker_model or "off",
    )
    yield


def _result_json(result: SearchResult) -> dict[str, Any]:
    node = result.hit.node
    return {
        "symbol": node.qualified_name,
        "file": node.file_path,
        "line": node.start_line,
        "score": round(result.hit.score, 4),
        "source": result.source,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Ripple", version="0.1.0", lifespan=lifespan)
    install_middleware(app)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        graph: CodeGraph | None = request.app.state.graph
        return {
            "status": "ok",
            "index": {
                **request.app.state.index_status,
                "nodes": graph.graph.number_of_nodes() if graph else 0,
                "edges": graph.graph.number_of_edges() if graph else 0,
            },
            "cache": await request.app.state.cache.stats(),
            "models": {
                "embedder": request.app.state.embedder.model_name,
                "reranker": settings.reranker_model or None,
            },
        }

    @app.get("/search")
    async def search(
        request: Request,
        q: str = Query(..., min_length=2, description="A 'where/how' question."),
        k: int = Query(5, ge=1, le=50),
        rerank: bool = Query(True),
        expand: bool = Query(True),
    ) -> JSONResponse:
        cache: ResponseCache = request.app.state.cache
        cache_key = f"search:{q}:{k}:{rerank}:{expand}"
        if (cached := await cache.get(cache_key)) is not None:
            payload = json.loads(cached)
            payload["meta"]["cache"] = "hit"
            return JSONResponse(payload)

        try:
            results, timings = await request.app.state.pipeline.search(
                q, k=k, rerank=rerank, expand=expand
            )
        except (OperationalError, ProgrammingError) as exc:
            raise HTTPException(503, "index database unreachable — is Postgres up?") from exc

        payload = {
            "query": q,
            "results": [_result_json(r) for r in results],
            "meta": {
                "timings_ms": timings,
                "reranked": rerank and request.app.state.reranker is not None,
                "expanded": expand,
                "cache": "miss",
            },
        }
        await cache.set(cache_key, json.dumps(payload))
        return JSONResponse(payload)

    @app.get("/impact")
    async def impact(request: Request, symbol: str = Query(..., min_length=1)) -> dict[str, Any]:
        graph: CodeGraph | None = request.app.state.graph
        if graph is None:
            raise HTTPException(503, "no index loaded — POST /index first")
        matches = graph.resolve_symbol(symbol)
        if not matches:
            raise HTTPException(404, f"symbol not found: {symbol!r}")
        if len(matches) > 1:
            raise HTTPException(400, f"ambiguous symbol {symbol!r}; candidates: {matches[:10]}")
        result = graph.impact(matches[0])
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

    @app.post("/index", status_code=202)
    async def index(
        request: Request, body: dict[str, Any], background: BackgroundTasks
    ) -> dict[str, Any]:
        repo_path = Path(str(body.get("repo_path", ""))).expanduser()
        full = bool(body.get("full", False))
        if not repo_path.is_dir():
            raise HTTPException(400, f"repo_path is not a directory: {repo_path}")
        if request.app.state.index_status.get("state") == "indexing":
            raise HTTPException(409, "an indexing run is already in progress")
        request.app.state.index_status = {"state": "indexing", "repo": str(repo_path)}
        background.add_task(_run_index, request.app, repo_path, full)
        return {"accepted": True, "repo_path": str(repo_path), "full": full}

    return app


async def _run_index(app: FastAPI, repo_path: Path, full: bool = False) -> None:
    """Background (re)index (incremental) -> swap the in-memory graph atomically."""
    try:
        graph, report = await asyncio.to_thread(
            index_repository, repo_path, app.state.embedder, full
        )
    except Exception as exc:  # noqa: BLE001 — status must reflect any failure
        logger.exception("indexing failed for %s", repo_path)
        app.state.index_status = {"state": "error", "repo": str(repo_path), "detail": str(exc)}
        return
    app.state.graph = graph
    app.state.pipeline.graph = graph
    await app.state.cache.flush()
    app.state.index_status = {
        "state": "ready",
        "repo": str(repo_path),
        "mode": report.mode,
        "files_changed": report.files_changed,
        "chunks_embedded": report.chunks_embedded,
        "timings_ms": report.timings_ms,
        "nodes": graph.graph.number_of_nodes(),
        "edges": graph.graph.number_of_edges(),
    }


app = create_app()

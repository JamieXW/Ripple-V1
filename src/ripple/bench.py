"""Async load generator for the Ripple API (M7).

Fires ``requests`` total requests at an endpoint with ``concurrency`` in flight,
then reports latency percentiles and throughput. Percentiles, not averages: the
tail (p95/p99) is what users feel under load, and averages hide it.

Scenarios:
- ``search``        — full pipeline, unique queries (cache-busting suffix) so every
                      request does real embed/retrieve/rerank work
- ``search-fast``   — same but ``rerank=false``
- ``search-cached`` — one repeated query: measures the Redis cache path
- ``impact``        — graph traversal endpoint

Run ``ripple serve`` first; results also land as JSON under benchmarks/results/.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

QUERIES = [
    "load the session from a signed cookie",
    "register a url route with the application",
    "serialize an object to a json response",
    "parse incoming request headers",
    "render a template with a context",
    "handle an http error and return a response",
    "configure the application from a file",
    "stream a file back to the client",
    "validate and decode form data",
    "redirect the user to another endpoint",
]


@dataclass
class BenchResult:
    scenario: str
    requests: int
    concurrency: int
    errors: int
    wall_s: float
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile over pre-sorted values (empty -> 0.0)."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def _request_params(
    scenario: str, i: int, impact_symbol: str, nonce: str = ""
) -> tuple[str, dict[str, Any]]:
    if scenario == "impact":
        return "/impact", {"symbol": impact_symbol}
    query = QUERIES[i % len(QUERIES)]
    if scenario == "search-cached":
        # Constant within a run (repeat hits), unique across runs (first is a real miss).
        return "/search", {"q": f"{query} {nonce}", "k": 5}
    # Per-run nonce + per-request index bust the cache within AND across runs —
    # without the nonce, a rerun silently serves Redis hits and reports fake speed.
    params: dict[str, Any] = {"q": f"{query} {nonce}{i}", "k": 5}
    if scenario == "search-fast":
        params["rerank"] = "false"
    return "/search", params


async def run_scenario(
    base_url: str,
    scenario: str,
    requests: int = 100,
    concurrency: int = 8,
    warmup: int = 3,
    impact_symbol: str = "flask.views.View",
) -> BenchResult:
    import uuid

    latencies: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)
    nonce = uuid.uuid4().hex[:6]

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        for i in range(warmup):
            path, params = _request_params(scenario, 10_000 + i, impact_symbol, nonce)
            await client.get(path, params=params)

        async def one(i: int) -> None:
            nonlocal errors
            path, params = _request_params(scenario, i, impact_symbol, nonce)
            async with semaphore:
                start = time.perf_counter()
                try:
                    response = await client.get(path, params=params)
                    if response.status_code != 200:
                        errors += 1
                        return
                except httpx.HTTPError:
                    errors += 1
                    return
                latencies.append((time.perf_counter() - start) * 1000)

        wall_start = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(requests)))
        wall = time.perf_counter() - wall_start

    latencies.sort()
    n = len(latencies)
    return BenchResult(
        scenario=scenario,
        requests=requests,
        concurrency=concurrency,
        errors=errors,
        wall_s=round(wall, 2),
        rps=round(n / wall, 1) if wall else 0.0,
        p50_ms=round(percentile(latencies, 50), 1),
        p95_ms=round(percentile(latencies, 95), 1),
        p99_ms=round(percentile(latencies, 99), 1),
        mean_ms=round(sum(latencies) / n, 1) if n else 0.0,
        max_ms=round(latencies[-1], 1) if n else 0.0,
    )

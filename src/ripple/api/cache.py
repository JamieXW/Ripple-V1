"""Redis response cache with hit/miss counters (M6).

Degrades gracefully: if Redis is unreachable at startup or any operation fails, the
cache turns itself off and every request just runs the pipeline — availability over
caching. Hit/miss counts live in Redis so the hit rate survives restarts and is
reported by ``/health`` (a measured cache hit rate is part of the systems story).
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_HITS = "ripple:cache:hits"
_MISSES = "ripple:cache:misses"


class ResponseCache:
    def __init__(self, url: str, ttl_seconds: int = 3600) -> None:
        self.url = url
        self.ttl = ttl_seconds
        self.available = False
        self._redis: Any = None

    async def connect(self) -> None:
        try:
            self._redis = aioredis.from_url(self.url, decode_responses=True)  # type: ignore[no-untyped-call]
            await self._redis.ping()
            self.available = True
        except Exception as exc:  # noqa: BLE001 — any failure means "run without cache"
            logger.warning("cache disabled (redis unreachable at %s): %s", self.url, exc)
            self.available = False

    async def get(self, key: str) -> str | None:
        if not self.available:
            return None
        try:
            value: str | None = await self._redis.get(f"ripple:resp:{key}")
            await self._redis.incr(_HITS if value is not None else _MISSES)
            return value
        except Exception:  # noqa: BLE001
            self.available = False
            return None

    async def set(self, key: str, value: str) -> None:
        if not self.available:
            return
        try:
            await self._redis.set(f"ripple:resp:{key}", value, ex=self.ttl)
        except Exception:  # noqa: BLE001
            self.available = False

    async def flush(self) -> None:
        """Drop all cached responses (called on reindex — answers may have changed)."""
        if not self.available:
            return
        try:
            async for key in self._redis.scan_iter("ripple:resp:*"):
                await self._redis.delete(key)
        except Exception:  # noqa: BLE001
            self.available = False

    async def stats(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False}
        try:
            hits = int(await self._redis.get(_HITS) or 0)
            misses = int(await self._redis.get(_MISSES) or 0)
            total = hits + misses
            return {
                "available": True,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 3) if total else None,
            }
        except Exception:  # noqa: BLE001
            self.available = False
            return {"available": False}

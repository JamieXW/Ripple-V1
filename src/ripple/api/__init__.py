"""API (M6): async FastAPI service exposing /index, /search, /impact, /health, with
Redis caching (hit rate reported), OpenTelemetry per-stage tracing, and the hybrid
retrieval pipeline (semantic seed -> graph expand -> rerank) behind /search."""

from ripple.api.app import app, create_app

__all__ = ["app", "create_app"]

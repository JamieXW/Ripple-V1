"""API (M6): async FastAPI service exposing /index, /search, /impact, /health, with
Redis caching and structured per-stage request tracing (parse -> retrieve -> graph-walk
-> rerank)."""

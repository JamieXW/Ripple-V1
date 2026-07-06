"""Environment-driven configuration (pydantic-settings).

All runtime config lives here so nothing is hardcoded. Override any field via an
environment variable prefixed with ``RIPPLE_`` (e.g. ``RIPPLE_DATABASE_URL=...``)
or a ``.env`` file in the working directory.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object. Fields gain real use as later milestones land."""

    model_config = SettingsConfigDict(
        env_prefix="RIPPLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage (M4). Host port 55432 to avoid clashing with other local Postgres
    # instances (this machine already has servers on 5432 and 5433).
    database_url: str = "postgresql+psycopg://ripple:ripple@localhost:55432/ripple"

    # Caching (M6). Host port 56379 for the same clash-avoidance reason as Postgres.
    redis_url: str = "redis://localhost:56379/0"
    cache_ttl_seconds: int = 3600

    # Retrieval pipeline (M6).
    reranker_model: str = ""  # path or HF name; empty = reranking off
    retrieve_k: int = 50  # candidates fetched before rerank
    expand_hops_cap: int = 20  # max graph-expansion candidates added

    # Inference devices + limits (M7, measured on Apple Silicon). Rule of thumb:
    # GPU wins real batches, CPU wins tiny ones (launch overhead) —
    #   single-query embed: cpu 3ms vs mps 6ms  -> query on CPU
    #   bulk index embed (512 chunks): mps 234ms vs cpu 484ms -> bulk auto (GPU)
    #   rerank 70 real-length pairs @256 tokens: mps 333ms vs cpu 546ms -> rerank auto
    # Truncating rerank inputs 512->256 tokens cut latency ~3x (attention cost is
    # superlinear in length; ranking signal concentrates early in a chunk).
    query_device: str = "cpu"  # single-query embedding at serve time
    rerank_device: str = ""  # cross-encoder device; empty = library auto (mps/cuda)
    rerank_max_length: int = 256  # token cap per (query, code) rerank pair
    bulk_embed_device: str = ""  # indexing; empty = library auto (mps/cuda if present)

    # Embeddings (M3). Tier-0 baseline; swapped for a code model / fine-tune later.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Observability (M6).
    otel_console: bool = False  # print OpenTelemetry spans to stdout

    # Logging.
    log_level: str = "INFO"


settings = Settings()

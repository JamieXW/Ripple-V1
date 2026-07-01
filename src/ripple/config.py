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

    # Caching (M6).
    redis_url: str = "redis://localhost:6379/0"

    # Embeddings (M3). Tier-0 baseline; swapped for a code model / fine-tune later.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Logging.
    log_level: str = "INFO"


settings = Settings()

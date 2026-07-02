"""Engine, session, and one-time schema setup for Postgres + pgvector."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session

from ripple.config import settings
from ripple.db.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide engine built from ``settings.database_url`` (cached)."""
    return create_engine(settings.database_url)


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    """Async engine for the service layer (psycopg's async mode, same URL)."""
    return create_async_engine(settings.database_url)


@asynccontextmanager
async def async_session_scope() -> AsyncIterator[AsyncSession]:
    """Async transactional session: commit on success, roll back on error."""
    session = AsyncSession(get_async_engine())
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commit on success, roll back on error."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    """Idempotently create the pgvector extension, tables, and the HNSW index."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Columns added after M4 for databases created before them (create_all only
        # creates missing tables, not missing columns; alembic once schema churn grows).
        conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_line INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content TEXT DEFAULT ''"))
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )

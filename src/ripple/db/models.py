"""SQLAlchemy ORM models for the persistent store (M4).

Three tables mirror the in-memory structures: ``nodes`` (definitions), ``edges``
(call/inherit relationships, keyed by qualified name), and ``chunks`` (one embedding
per function/class in a pgvector ``vector`` column). ``file_hashes`` (M7) and a test
``coverage`` table are deferred until their milestones have a consumer.
"""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: Embedding dimension for all-MiniLM-L6-v2. Fixed at table-creation time.
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class NodeRow(Base):
    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    qualified_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    kind: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    start_line: Mapped[int] = mapped_column()
    end_line: Mapped[int] = mapped_column()
    docstring: Mapped[str | None] = mapped_column(String, nullable=True)


class EdgeRow(Base):
    __tablename__ = "edges"

    id: Mapped[int] = mapped_column(primary_key=True)
    src: Mapped[str] = mapped_column(String, index=True)
    dst: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    qualified_name: Mapped[str] = mapped_column(String, index=True)
    file_path: Mapped[str] = mapped_column(String)
    start_line: Mapped[int] = mapped_column()
    model_name: Mapped[str] = mapped_column(String)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))

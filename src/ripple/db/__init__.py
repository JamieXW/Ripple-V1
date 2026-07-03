"""Storage (M4): Postgres + pgvector. SQLAlchemy models, session/engine, schema
setup, and read/write of the graph + vector index."""

from ripple.db.models import Base, ChunkRow, EdgeRow, FileHashRow, NodeRow
from ripple.db.repository import (
    apply_incremental,
    load_file_hashes,
    load_graph_from_db,
    replace_graph,
    search_chunks,
    stored_model_name,
    write_index,
)
from ripple.db.session import get_engine, init_db, session_scope

__all__ = [
    "Base",
    "ChunkRow",
    "EdgeRow",
    "FileHashRow",
    "NodeRow",
    "apply_incremental",
    "get_engine",
    "init_db",
    "load_file_hashes",
    "load_graph_from_db",
    "replace_graph",
    "search_chunks",
    "session_scope",
    "stored_model_name",
    "write_index",
]

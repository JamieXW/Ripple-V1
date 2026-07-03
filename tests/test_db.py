"""Postgres + pgvector round-trip tests.

Each test runs inside a transaction that is rolled back, so it never persists to or
clobbers a real index. Skipped automatically when Postgres isn't reachable.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ripple.db import get_engine, init_db, load_graph_from_db, search_chunks, write_index
from ripple.db.models import EMBEDDING_DIM
from ripple.embeddings import VectorIndex
from ripple.graph import build_graph
from ripple.parsing import parse_source
from ripple.parsing.models import CodeNode


@pytest.fixture
def session() -> Iterator[Session]:
    engine = get_engine()
    try:
        init_db(engine)
    except OperationalError:
        pytest.skip("Postgres not reachable (run: docker compose up -d)")
    db = Session(engine)
    try:
        yield db
    finally:
        db.rollback()  # discard everything this test wrote
        db.close()


def _two_function_graph() -> object:
    module = parse_source(
        "def a():\n    return b()\n\ndef b():\n    return 1\n", "pkg.m", "pkg/m.py"
    )
    return build_graph([module])


def _fake_vectors() -> VectorIndex:
    nodes = [
        CodeNode("pkg.m.a", "function", "pkg/m.py", 1, 2, None),
        CodeNode("pkg.m.b", "function", "pkg/m.py", 3, 4, None),
    ]
    matrix = np.zeros((2, EMBEDDING_DIM), dtype=np.float32)
    matrix[0, 0] = 1.0  # 'a' points along axis 0
    matrix[1, 1] = 1.0  # 'b' points along axis 1
    return VectorIndex(nodes=nodes, matrix=matrix, model_name="fake-384")


def test_write_then_load_graph_preserves_impact(session: Session) -> None:
    write_index(session, _two_function_graph(), _fake_vectors())  # type: ignore[arg-type]
    session.flush()
    loaded = load_graph_from_db(session)
    assert "pkg.m.b" in loaded.nodes
    affected = {a.node.qualified_name for a in loaded.impact("pkg.m.b").affected}
    assert "pkg.m.a" in affected  # a -> b survived the round trip


def test_search_returns_nearest_vector(session: Session) -> None:
    write_index(session, _two_function_graph(), _fake_vectors())  # type: ignore[arg-type]
    session.flush()
    query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0  # closest to 'a'
    results = search_chunks(session, query, k=1)
    assert results[0].node.qualified_name == "pkg.m.a"
    assert results[0].score > 0.99  # cosine similarity ~1.0


def test_apply_incremental_touches_only_changed_chunks(session: Session) -> None:
    from sqlalchemy import select

    from ripple.db import apply_incremental, load_file_hashes
    from ripple.db.models import ChunkRow

    graph = _two_function_graph()
    write_index(session, graph, _fake_vectors(), {"pkg/m.py": "hash-v1"})  # type: ignore[arg-type]
    session.flush()

    # Pretend pkg/m.py changed and its re-embed produced a single chunk for 'b'.
    changed = VectorIndex(
        nodes=[CodeNode("pkg.m.b", "function", "pkg/m.py", 3, 4, None)],
        matrix=np.full((1, EMBEDDING_DIM), 0.5, dtype=np.float32),
        model_name="fake-384",
        texts=["def b(): return 2"],
    )
    apply_incremental(
        session,
        graph,  # type: ignore[arg-type]
        changed,
        frozenset({"pkg/m.py"}),
        frozenset(),
        {"pkg/m.py": "hash-v2"},
    )
    session.flush()

    rows = session.execute(select(ChunkRow)).scalars().all()
    assert [r.qualified_name for r in rows] == ["pkg.m.b"]  # stale chunks replaced
    assert rows[0].content == "def b(): return 2"
    assert load_file_hashes(session) == {"pkg/m.py": "hash-v2"}

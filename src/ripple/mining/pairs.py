"""Docstring-mined (query, code) pairs — free supervision for search eval and training.

A function's docstring is a natural-language description of it, so
``(docstring, docstring-stripped source)`` is a labelled (query, relevant-code) pair with
zero manual annotation. The docstring MUST be stripped from the code side: leaving it in
lets a retriever match the query against its own text verbatim (leakage), inflating every
metric — same principle as never training on the test set.

Pairs are split train/test deterministically by a hash of the qualified name, so the
held-out set is stable across runs and machines; M5b trains only on the train split and
the eval only ever queries the test split.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path

from ripple.embeddings.vector_index import iter_chunks
from ripple.parsing.models import ParsedModule

#: Docstrings shorter than this are too vague to act as a query ("TODO", "Helper.").
MIN_QUERY_CHARS = 20

Split = str  # "train" | "test"


@dataclass(frozen=True)
class QueryCodePair:
    """One mined example: this natural-language query should retrieve this code."""

    qualified_name: str
    file_path: str
    start_line: int
    query: str  # the docstring
    code: str  # source snippet with the docstring removed
    split: Split


def split_for(qualified_name: str, test_fraction: float = 0.2) -> Split:
    """Deterministic train/test assignment (md5, not ``hash()``, which is salted)."""
    digest = hashlib.md5(qualified_name.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "test" if bucket < int(test_fraction * 100) else "train"


def strip_docstring(snippet: str) -> str | None:
    """Return ``snippet`` with its docstring removed, or ``None`` if that leaves nothing.

    ``None`` also covers snippets we cannot parse (unusual layouts) — callers skip those
    rather than risk leaking the docstring into the code side.
    """
    dedented = textwrap.dedent(snippet)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return None
    body = node.body
    has_docstring = (
        bool(body)
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )
    if not has_docstring:
        return dedented
    if len(body) == 1:
        return None  # the docstring was the whole body — nothing left to retrieve on
    doc = body[0]
    lines = dedented.splitlines()
    del lines[doc.lineno - 1 : (doc.end_lineno or doc.lineno)]
    return "\n".join(lines)


def mine_docstring_pairs(modules: list[ParsedModule], repo_root: Path) -> list[QueryCodePair]:
    """Extract every usable (docstring, stripped-code) pair from the parsed repo."""
    pairs: list[QueryCodePair] = []
    for node, snippet in iter_chunks(modules, repo_root):
        if not node.docstring or len(node.docstring) < MIN_QUERY_CHARS:
            continue
        code = strip_docstring(snippet)
        if code is None or not code.strip():
            continue
        pairs.append(
            QueryCodePair(
                qualified_name=node.qualified_name,
                file_path=node.file_path,
                start_line=node.start_line,
                query=node.docstring,
                code=code,
                split=split_for(node.qualified_name),
            )
        )
    return pairs

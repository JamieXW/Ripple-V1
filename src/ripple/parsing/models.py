"""Data structures produced by parsing, consumed by the graph builder.

A single AST walk yields one :class:`ParsedModule` per file. Definitions become
:class:`CodeNode` records (the graph's nodes); calls and base classes are captured
as *raw* references (names as written in source) to be resolved to real nodes later
by the resolver — name resolution is a separate, harder step (see graph/resolver.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Kinds of code unit we track as graph nodes.
NodeKind = str  # one of: "module", "class", "function"


@dataclass(frozen=True)
class CodeNode:
    """A definition in the source: a module, class, or function/method."""

    qualified_name: str
    kind: NodeKind
    file_path: str  # relative to the repo root
    start_line: int
    end_line: int
    docstring: str | None = None


@dataclass(frozen=True)
class RawCall:
    """An unresolved call site: ``caller`` invokes something written as ``callee``."""

    caller: str  # qualified name of the enclosing definition (function or module)
    callee: str  # dotted name as written, e.g. "validate" or "self.save" or "db.commit"
    lineno: int


@dataclass(frozen=True)
class RawInherit:
    """An unresolved base class: ``subclass`` inherits from something named ``base``."""

    subclass: str  # qualified name of the subclass
    base: str  # base name as written, e.g. "Base" or "ext.Model"
    lineno: int


@dataclass
class ParsedModule:
    """Everything extracted from one source file."""

    module_qname: str
    file_path: str
    nodes: list[CodeNode] = field(default_factory=list)
    calls: list[RawCall] = field(default_factory=list)
    inherits: list[RawInherit] = field(default_factory=list)
    # Local alias -> dotted import target, e.g. {"np": "numpy", "Flask": "flask.app.Flask"}.
    imports: dict[str, str] = field(default_factory=dict)

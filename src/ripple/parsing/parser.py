"""AST parsing: source files -> :class:`ParsedModule` records.

The :class:`_ModuleVisitor` walks one file's syntax tree, tracking a scope stack so
each definition gets a fully qualified name (``pkg.mod.Class.method``). Calls are
attributed to the nearest enclosing function (or the module for top-level code).
Call targets and base classes are recorded as written; resolving them to real nodes
happens later in the resolver.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterator
from pathlib import Path

from ripple.parsing.models import CodeNode, ParsedModule, RawCall, RawInherit

logger = logging.getLogger(__name__)

#: Directories we never descend into when discovering source files.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        ".tox",
    }
)


def iter_python_files(repo_root: Path) -> Iterator[Path]:
    """Yield every ``.py`` file under ``repo_root``, skipping vendored/cache dirs."""
    for path in sorted(repo_root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.relative_to(repo_root).parts):
            continue
        yield path


def module_qname_from_path(path: Path, repo_root: Path) -> str:
    """Derive a module's qualified name from its path (heuristic).

    Drops a leading ``src/`` (src-layout), strips ``.py``, and collapses
    ``__init__`` to its package. Good enough for intra-repo resolution; documented
    as imperfect for unusual layouts.
    """
    parts = list(path.relative_to(repo_root).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _dotted_name(node: ast.expr) -> str | None:
    """Flatten a call target into a dotted string: ``a.b.c`` -> ``"a.b.c"``.

    Returns ``None`` when the base isn't a plain name/attribute chain (e.g. the
    result of another call or a subscript), which we can't resolve statically.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


class _ModuleVisitor(ast.NodeVisitor):
    """Single-pass collector for one module's nodes, calls, inherits, and imports."""

    def __init__(self, module_qname: str, file_path: str) -> None:
        self.module_qname = module_qname
        self.file_path = file_path
        self.nodes: list[CodeNode] = []
        self.calls: list[RawCall] = []
        self.inherits: list[RawInherit] = []
        self.imports: dict[str, str] = {}
        self._name_stack: list[str] = []  # class/function names for qualified naming
        self._func_owner: list[str] = []  # enclosing function qnames (call attribution)

    def _qual(self, name: str) -> str:
        return ".".join([self.module_qname, *self._name_stack, name])

    @property
    def _owner(self) -> str:
        return self._func_owner[-1] if self._func_owner else self.module_qname

    def parse_module(self, tree: ast.Module) -> None:
        end = max((getattr(n, "end_lineno", 1) or 1) for n in ast.walk(tree)) if tree.body else 1
        self.nodes.append(
            CodeNode(
                qualified_name=self.module_qname,
                kind="module",
                file_path=self.file_path,
                start_line=1,
                end_line=end,
                docstring=ast.get_docstring(tree),
            )
        )
        self.generic_visit(tree)

    def _visit_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        qname = self._qual(node.name)
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        self.nodes.append(
            CodeNode(
                qualified_name=qname,
                kind=kind,
                file_path=self.file_path,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                docstring=ast.get_docstring(node),
            )
        )
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = _dotted_name(base)
                if base_name:
                    self.inherits.append(
                        RawInherit(subclass=qname, base=base_name, lineno=node.lineno)
                    )

        self._name_stack.append(node.name)
        if kind == "function":
            self._func_owner.append(qname)
        self.generic_visit(node)
        if kind == "function":
            self._func_owner.pop()
        self._name_stack.pop()

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def
    visit_ClassDef = _visit_def

    def visit_Call(self, node: ast.Call) -> None:
        callee = _dotted_name(node.func)
        if callee:
            self.calls.append(RawCall(caller=self._owner, callee=callee, lineno=node.lineno))
        self.generic_visit(node)  # nested calls live in args/keywords

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.imports[alias.asname] = alias.name
            else:
                top = alias.name.split(".")[0]
                self.imports[top] = top
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Relative import (level > 0): resolve against the current package.
        base_parts = self.module_qname.split(".")[: -node.level] if node.level else []
        prefix_parts = [*base_parts, node.module] if node.module else base_parts
        prefix = ".".join(p for p in prefix_parts if p)
        for alias in node.names:
            local = alias.asname or alias.name
            target = f"{prefix}.{alias.name}" if prefix else alias.name
            self.imports[local] = target
        self.generic_visit(node)


def parse_source(source: str, module_qname: str, file_path: str) -> ParsedModule:
    """Parse a source string into a :class:`ParsedModule`."""
    visitor = _ModuleVisitor(module_qname, file_path)
    visitor.parse_module(ast.parse(source))
    return ParsedModule(
        module_qname=module_qname,
        file_path=file_path,
        nodes=visitor.nodes,
        calls=visitor.calls,
        inherits=visitor.inherits,
        imports=visitor.imports,
    )


def parse_file(path: Path, repo_root: Path) -> ParsedModule | None:
    """Parse one file. Returns ``None`` (and logs) on read/syntax errors."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("skipping unreadable file %s: %s", path, exc)
        return None
    module_qname = module_qname_from_path(path, repo_root)
    file_path = str(path.relative_to(repo_root))
    try:
        return parse_source(source, module_qname, file_path)
    except SyntaxError as exc:
        logger.warning("skipping file with syntax error %s: %s", path, exc)
        return None


def parse_repo(repo_root: Path) -> list[ParsedModule]:
    """Parse every discoverable Python file under ``repo_root``."""
    modules = [parse_file(path, repo_root) for path in iter_python_files(repo_root)]
    return [m for m in modules if m is not None]

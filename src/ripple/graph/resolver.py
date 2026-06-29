"""Name resolution: turn raw call/inherit references into graph edges.

This is the hard, lossy step. A reference written as ``validate`` or ``self.save``
or ``db.commit`` must be mapped to the actual definition it targets — across imports,
class scope, and the whole repo. We resolve what's statically clear and *count* what
we can't (see :class:`ResolutionStats`), favouring precision: when unsure, we drop the
edge rather than invent a wrong one. Known gaps (dynamic dispatch, inherited-method
calls, instance-method calls via local variables) are intentional and measured in M2.
"""

from __future__ import annotations

from collections import defaultdict

from ripple.graph.models import Edge, ResolutionStats
from ripple.parsing.models import CodeNode, ParsedModule


class _Resolver:
    def __init__(self, modules: list[ParsedModule]) -> None:
        self.modules = modules
        # qualified name -> node (every definition, including modules)
        self.node_index: dict[str, CodeNode] = {
            node.qualified_name: node for module in modules for node in module.nodes
        }
        # short name -> set of qualified names (functions & classes only) for fallback
        self.by_short: dict[str, set[str]] = defaultdict(set)
        for qname, node in self.node_index.items():
            if node.kind in ("function", "class"):
                self.by_short[qname.rsplit(".", 1)[-1]].add(qname)

    def _enclosing_class(self, caller: str) -> str | None:
        """The class qname a method belongs to, if ``caller`` is a method."""
        parent = caller.rsplit(".", 1)[0] if "." in caller else None
        if parent and self.node_index.get(parent) and self.node_index[parent].kind == "class":
            return parent
        return None

    def _resolve(self, name: str, module: ParsedModule, enclosing_class: str | None) -> str:
        """Return a target qname, or a ``"!<reason>"`` sentinel if unresolved."""
        parts = name.split(".")
        head = parts[0]

        # 1. self/cls method or attribute on the enclosing class.
        if head in ("self", "cls") and enclosing_class and len(parts) >= 2:
            candidate = f"{enclosing_class}.{parts[1]}"
            return candidate if candidate in self.node_index else "!self_miss"

        # 2. A definition in the same module (local defs shadow imports in practice).
        same_module = f"{module.module_qname}.{name}"
        if same_module in self.node_index:
            return same_module

        # 3. An imported name: rewrite through the import alias, then look up.
        if head in module.imports:
            candidate = ".".join([module.imports[head], *parts[1:]])
            # Resolves only if the import points back into this repo; otherwise external.
            return candidate if candidate in self.node_index else "!external"

        # 4. Last-resort: a bare (undotted) name that matches exactly one definition.
        if len(parts) == 1:
            matches = self.by_short.get(name)
            if matches and len(matches) == 1:
                return next(iter(matches))
            if matches and len(matches) > 1:
                return "!ambiguous"

        # A dotted base we can't trace (local variable, dynamic attribute, etc.).
        return "!unknown"

    def resolve(self) -> tuple[list[Edge], ResolutionStats]:
        edges: set[Edge] = set()
        stats = ResolutionStats()

        def record(target: str, src: str, kind: str) -> None:
            if not target.startswith("!"):
                stats.resolved += 1
                if target != src:  # skip trivial self-loops
                    edges.add(Edge(src=src, dst=target, kind=kind))
                return
            reason = target[1:]
            stats.by_reason[reason] = stats.by_reason.get(reason, 0) + 1
            if reason == "external":
                stats.external += 1
            elif reason == "ambiguous":
                stats.ambiguous += 1
            elif reason == "self_miss":
                stats.self_miss += 1
            else:
                stats.unknown += 1

        for module in self.modules:
            for call in module.calls:
                target = self._resolve(call.callee, module, self._enclosing_class(call.caller))
                record(target, call.caller, "calls")
            for inh in module.inherits:
                target = self._resolve(inh.base, module, None)
                record(target, inh.subclass, "inherits")

        return sorted(edges, key=lambda e: (e.src, e.dst, e.kind)), stats


def resolve_modules(modules: list[ParsedModule]) -> tuple[list[Edge], ResolutionStats]:
    """Resolve all raw references in ``modules`` to graph edges, with diagnostics."""
    return _Resolver(modules).resolve()

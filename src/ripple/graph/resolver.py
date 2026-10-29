"""Name resolution: turn raw call/inherit references into graph edges.

This is the hard, lossy step. A reference written as ``validate`` or ``self.save``
or ``db.commit`` must be mapped to the actual definition it targets — across imports,
class scope, and the whole repo. We resolve what's statically clear and *count* what
we can't (see :class:`ResolutionStats`), favouring precision: when unsure, we drop the
edge rather than invent a wrong one. Known gaps (dynamic dispatch, instance-method
calls via local variables) are intentional and measured in M2.

Resolution runs in two passes. Inherits references resolve first, giving each class
its parent classes by qualified name; calls resolve second, so ``self.x`` / ``cls.x`` /
``super().x`` can walk that inheritance chain in Python's own method-resolution order
(the class, then each base depth-first, left to right) — resolution v2, lever 1.
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
        # class qname -> resolved in-repo base class qnames, in declaration order.
        # Filled by the inherits pass; consulted by the call pass (MRO walk).
        self.bases: dict[str, list[str]] = defaultdict(list)
        self._mro_cache: dict[str, list[str]] = {}

    def _mro(self, cls: str) -> list[str]:
        """Python's method resolution order for ``cls`` (C3 linearization).

        Only in-repo bases take part: an external parent was never resolved, so the
        chain simply ends there. If a hierarchy is inconsistent (C3 has no answer —
        Python itself would raise TypeError) we fall back to a plain depth-first order.
        """
        cached = self._mro_cache.get(cls)
        if cached is not None:
            return cached
        self._mro_cache[cls] = [cls]  # provisional entry breaks import-time cycles
        bases = [b for b in self.bases.get(cls, []) if b != cls]
        seqs = [self._mro(b)[:] for b in bases] + [bases[:]]
        result = [cls]
        while any(seqs):
            head = next(
                (s[0] for s in seqs if s and not any(s[0] in t[1:] for t in seqs)),
                None,
            )
            if head is None:  # inconsistent hierarchy: degrade gracefully
                result.extend(_dfs_order(cls, self.bases)[1:])
                break
            result.append(head)
            seqs = [s[1:] if s and s[0] == head else s for s in seqs]
        self._mro_cache[cls] = result
        return result

    def _lookup_in_mro(self, cls: str, attr: str, *, skip_self: bool = False) -> str | None:
        """Find ``attr`` on ``cls`` or its ancestors, in Python's lookup order.

        ``skip_self`` models ``super().attr``: start one step past the class itself.
        We never guess at what an external parent defines.
        """
        order = self._mro(cls)
        for current in order[1:] if skip_self else order:
            candidate = f"{current}.{attr}"
            if candidate in self.node_index:
                return candidate
        return None

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

        # 1. self/cls method on the enclosing class or an ancestor; super().x starts
        #    the same walk one level up.
        if head in ("self", "cls", "super") and enclosing_class and len(parts) >= 2:
            found = self._lookup_in_mro(enclosing_class, parts[1], skip_self=head == "super")
            if found:
                # For ``self.jinja_env.get_template()`` this links the property/method
                # ``jinja_env`` — the dependency we can see. The method on its *value*
                # needs type inference (lever 2).
                return found
            # ``self.url_map.bind``: a method on an attribute's value. Counted apart
            # from plain misses so lever 2's target stays visible in the stats.
            return "!self_chain" if len(parts) >= 3 else "!self_miss"

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

        # Pass 1: inheritance, so the call pass can walk each class's ancestors.
        for module in self.modules:
            for inh in module.inherits:
                target = self._resolve(inh.base, module, None)
                record(target, inh.subclass, "inherits")
                if not target.startswith("!"):
                    self.bases[inh.subclass].append(target)
        # Pass 2: calls.
        for module in self.modules:
            for call in module.calls:
                target = self._resolve(call.callee, module, self._enclosing_class(call.caller))
                record(target, call.caller, "calls")

        return sorted(edges, key=lambda e: (e.src, e.dst, e.kind)), stats


def _dfs_order(cls: str, bases: dict[str, list[str]]) -> list[str]:
    """Depth-first, left-to-right ancestor order with cycle protection (C3 fallback)."""
    order: list[str] = []
    seen: set[str] = set()
    stack = [cls]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        order.append(current)
        stack.extend(reversed(bases.get(current, [])))
    return order


def resolve_modules(modules: list[ParsedModule]) -> tuple[list[Edge], ResolutionStats]:
    """Resolve all raw references in ``modules`` to graph edges, with diagnostics."""
    return _Resolver(modules).resolve()

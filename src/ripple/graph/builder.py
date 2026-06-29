"""Build the dependency graph and answer change-impact queries.

Edges point from depender to dependency (``A -> B`` means "A calls/inherits B"), so
the blast radius of changing ``B`` — everyone who transitively depends on it — is
exactly the set of nodes with a path *to* ``B`` (``networkx.ancestors``). That single
traversal is the differentiating feature of the whole project.
"""

from __future__ import annotations

import networkx as nx

from ripple.graph.models import AffectedNode, ImpactResult, ResolutionStats
from ripple.graph.resolver import resolve_modules
from ripple.parsing.models import CodeNode, ParsedModule


class CodeGraph:
    """An in-memory dependency graph over a repo's definitions."""

    def __init__(
        self,
        graph: nx.DiGraph[str],
        nodes: dict[str, CodeNode],
        stats: ResolutionStats,
        repo_root: str = "",
    ) -> None:
        self.graph = graph
        self.nodes = nodes
        self.stats = stats
        self.repo_root = repo_root

    @classmethod
    def from_modules(cls, modules: list[ParsedModule], repo_root: str = "") -> CodeGraph:
        nodes = {node.qualified_name: node for module in modules for node in module.nodes}
        edges, stats = resolve_modules(modules)
        graph: nx.DiGraph[str] = nx.DiGraph()
        for qname in nodes:
            graph.add_node(qname)
        for edge in edges:
            # Both endpoints must be real definitions in the repo.
            if edge.src in nodes and edge.dst in nodes:
                graph.add_edge(edge.src, edge.dst, kind=edge.kind)
        return cls(graph, nodes, stats, repo_root)

    def resolve_symbol(self, symbol: str) -> list[str]:
        """Find qualified names matching a user-supplied symbol.

        Matches an exact qualified name, or any node whose name ends in ``.symbol``
        (so ``save``, ``Admin.save``, and ``pkg.auth.Admin.save`` all work).
        """
        if symbol in self.nodes:
            return [symbol]
        suffix = f".{symbol}"
        return sorted(q for q in self.nodes if q == symbol or q.endswith(suffix))

    def impact(self, target: str) -> ImpactResult:
        """Compute the blast radius of changing ``target`` (an exact qualified name)."""
        if target not in self.nodes:
            raise KeyError(target)
        ancestors = nx.ancestors(self.graph, target) if target in self.graph else set()
        direct = set(self.graph.predecessors(target)) if target in self.graph else set()
        affected = [
            AffectedNode(node=self.nodes[q], direct=q in direct)
            for q in ancestors
            if q in self.nodes
        ]
        # Direct callers first, then by location for stable, readable output.
        affected.sort(key=lambda a: (not a.direct, a.node.file_path, a.node.start_line))
        return ImpactResult(target=self.nodes[target], affected=affected)

    def __repr__(self) -> str:
        return (
            f"CodeGraph(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )


def build_graph(modules: list[ParsedModule], repo_root: str = "") -> CodeGraph:
    """Convenience: resolve references and build a :class:`CodeGraph` from modules."""
    return CodeGraph.from_modules(modules, repo_root)

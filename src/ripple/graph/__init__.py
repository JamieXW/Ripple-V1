"""Graph (M1): build the call/import/inherit graph and traverse it to answer
change-impact questions (transitive callers/dependents = the blast radius).
In-memory (NetworkX) first; Postgres edge tables in M4."""

from ripple.graph.builder import CodeGraph, build_graph
from ripple.graph.models import AffectedNode, Edge, ImpactResult, ResolutionStats
from ripple.graph.resolver import resolve_modules

__all__ = [
    "AffectedNode",
    "CodeGraph",
    "Edge",
    "ImpactResult",
    "ResolutionStats",
    "build_graph",
    "resolve_modules",
]

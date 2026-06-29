"""Graph-layer data structures: resolved edges, resolution diagnostics, results."""

from __future__ import annotations

from dataclasses import dataclass, field

from ripple.parsing.models import CodeNode

#: Edge kinds in the dependency graph.
EdgeKind = str  # "calls" | "inherits"


@dataclass(frozen=True)
class Edge:
    """A resolved relationship: ``src`` calls / inherits from ``dst``."""

    src: str
    dst: str
    kind: EdgeKind


@dataclass
class ResolutionStats:
    """How resolution went — the honest record of what we could and couldn't link.

    These counts feed the limitations we report (CLAUDE.md §14) and the eval in M2.
    """

    resolved: int = 0
    # References we deliberately did not link, by reason:
    external: int = 0  # target lives outside the repo (stdlib / third-party / import miss)
    ambiguous: int = 0  # bare name matched multiple definitions
    unknown: int = 0  # dotted base we couldn't statically resolve (e.g. local var method)
    self_miss: int = 0  # self/cls attribute not found on the enclosing class
    by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.resolved + self.external + self.ambiguous + self.unknown + self.self_miss

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0


@dataclass(frozen=True)
class AffectedNode:
    """A node in the blast radius, with whether it calls the target directly."""

    node: CodeNode
    direct: bool  # True if it calls/inherits the target itself (vs. transitively)


@dataclass
class ImpactResult:
    """The blast radius of changing ``target``: who transitively depends on it."""

    target: CodeNode
    affected: list[AffectedNode]

    @property
    def direct_count(self) -> int:
        return sum(1 for a in self.affected if a.direct)

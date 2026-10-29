"""Graph tests: name resolution edge cases and impact (blast-radius) traversal."""

from __future__ import annotations

from ripple.graph import build_graph
from ripple.parsing import parse_source

# A small package with a known call structure, reused across tests:
#   auth.login        -> utils.validate (imported), auth.create_session (same module)
#   utils.validate    -> utils.check_token (same module), bool (builtin -> unresolved)
#   admin.admin_login -> auth.login (imported)
AUTH = parse_source(
    "from sample.utils import validate\n"
    "def login(user):\n"
    "    return validate(user) and create_session(user)\n"
    "def create_session(user):\n"
    "    return {'user': user}\n",
    "sample.auth",
    "sample/auth.py",
)
UTILS = parse_source(
    "def validate(user):\n"
    "    return bool(user) and check_token(user)\n"
    "def check_token(user):\n"
    "    return True\n",
    "sample.utils",
    "sample/utils.py",
)
ADMIN = parse_source(
    "from sample.auth import login\ndef admin_login(user):\n    return login(user)\n",
    "sample.admin",
    "sample/admin.py",
)


def test_self_method_call_resolves_to_class_method_not_module_function() -> None:
    module = parse_source(
        "def save():\n    pass\n"
        "class A:\n"
        "    def run(self):\n"
        "        return self.save()\n"
        "    def save(self):\n"
        "        return save()\n",
        "m",
        "m.py",
    )
    edges = set(build_graph([module]).graph.edges())
    assert ("m.A.run", "m.A.save") in edges  # self.save -> the method
    assert ("m.A.save", "m.save") in edges  # bare save() -> the module function


def test_imported_call_and_inheritance_resolve_across_modules() -> None:
    base = parse_source("class Base:\n    pass\n", "sample.base", "sample/base.py")
    sub = parse_source(
        "from sample.base import Base\nclass Sub(Base):\n    pass\n",
        "sample.models",
        "sample/models.py",
    )
    edges = {(u, v, d["kind"]) for u, v, d in build_graph([base, sub]).graph.edges(data=True)}
    assert ("sample.models.Sub", "sample.base.Base", "inherits") in edges


def test_transitive_blast_radius() -> None:
    graph = build_graph([AUTH, UTILS, ADMIN])
    result = graph.impact("sample.utils.check_token")
    names = {a.node.qualified_name for a in result.affected}
    assert names == {"sample.utils.validate", "sample.auth.login", "sample.admin.admin_login"}
    direct = {a.node.qualified_name for a in result.affected if a.direct}
    assert direct == {"sample.utils.validate"}


def test_leaf_target_has_empty_blast_radius() -> None:
    graph = build_graph([AUTH, UTILS, ADMIN])
    assert graph.impact("sample.admin.admin_login").affected == []


def test_builtin_calls_are_unresolved_not_edges() -> None:
    graph = build_graph([AUTH, UTILS, ADMIN])
    assert graph.stats.unknown >= 1  # bool(...) cannot resolve to a repo node
    assert graph.stats.resolution_rate > 0.5


def test_resolve_symbol_reports_ambiguity() -> None:
    one = parse_source("def save():\n    pass\n", "a", "a.py")
    two = parse_source("def save():\n    pass\n", "b", "b.py")
    matches = build_graph([one, two]).resolve_symbol("save")
    assert matches == ["a.save", "b.save"]


def test_resolve_symbol_matches_partial_qualified_name() -> None:
    module = parse_source("class A:\n    def save(self):\n        pass\n", "m", "m.py")
    assert build_graph([module]).resolve_symbol("A.save") == ["m.A.save"]


def test_self_call_resolves_through_parent_and_grandparent() -> None:
    base = parse_source(
        "class Base:\n    def save(self):\n        pass\n", "pkg.base", "pkg/base.py"
    )
    mid = parse_source(
        "from pkg.base import Base\nclass Mid(Base):\n    def load(self):\n        pass\n",
        "pkg.mid",
        "pkg/mid.py",
    )
    leaf = parse_source(
        "from pkg.mid import Mid\n"
        "class Leaf(Mid):\n"
        "    def run(self):\n"
        "        self.load()\n"  # defined on the parent
        "        self.save()\n"  # defined on the grandparent
        "        return cls_helper()\n"
        "def cls_helper():\n    pass\n",
        "pkg.leaf",
        "pkg/leaf.py",
    )
    graph = build_graph([base, mid, leaf])
    edges = set(graph.graph.edges())
    assert ("pkg.leaf.Leaf.run", "pkg.mid.Mid.load") in edges
    assert ("pkg.leaf.Leaf.run", "pkg.base.Base.save") in edges
    assert graph.stats.self_miss == 0
    # And impact sees it: changing Base.save now reaches Leaf.run transitively.
    affected = {a.node.qualified_name for a in graph.impact("pkg.base.Base.save").affected}
    assert "pkg.leaf.Leaf.run" in affected


def test_mro_prefers_nearest_definition_in_python_order() -> None:
    module = parse_source(
        "class A:\n    def go(self):\n        pass\n"
        "class B(A):\n    def go(self):\n        pass\n"
        "class C(A):\n    pass\n"
        "class D(C, B):\n"
        "    def run(self):\n"
        "        return self.go()\n",
        "m",
        "m.py",
    )
    edges = set(build_graph([module]).graph.edges())
    # Python's MRO for D is D, C, B, A — the override on B wins over A.
    assert ("m.D.run", "m.B.go") in edges
    assert ("m.D.run", "m.A.go") not in edges


def test_super_call_starts_lookup_at_parent_and_skips_override() -> None:
    module = parse_source(
        "class A:\n    def save(self):\n        pass\n"
        "class B(A):\n"
        "    def save(self):\n"
        "        return super().save()\n",
        "m",
        "m.py",
    )
    graph = build_graph([module])
    assert ("m.B.save", "m.A.save") in set(graph.graph.edges())
    assert graph.stats.ambiguous == 0  # super().save used to look like a bare 'save'


def test_external_parent_stops_the_walk_without_guessing() -> None:
    module = parse_source(
        "import click\n"
        "class Group(click.Group):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.unknown_method()\n"
        "    def run(self):\n"
        "        return self.app.route()\n",
        "m",
        "m.py",
    )
    graph = build_graph([module])
    assert graph.graph.number_of_edges() == 0
    assert graph.stats.self_miss == 2  # super().__init__ and self.unknown_method
    assert graph.stats.by_reason.get("self_chain") == 1  # self.app.route


def test_inheritance_cycle_does_not_hang() -> None:
    module = parse_source(
        "class A(B):\n    def run(self):\n        return self.missing()\n"
        "class B(A):\n    pass\n",
        "m",
        "m.py",
    )
    graph = build_graph([module])
    assert graph.stats.self_miss == 1

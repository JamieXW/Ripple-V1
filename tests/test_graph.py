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

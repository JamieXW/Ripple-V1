"""Parser tests: definitions, qualified names, calls, inheritance, imports."""

from __future__ import annotations

from pathlib import Path

from ripple.parsing import parse_source
from ripple.parsing.parser import parse_file


def test_extracts_definitions_with_qualified_names() -> None:
    module = parse_source(
        "def f():\n    pass\n\nclass C:\n    def g(self):\n        pass\n",
        "pkg.mod",
        "pkg/mod.py",
    )
    kinds = {node.qualified_name: node.kind for node in module.nodes}
    assert kinds == {
        "pkg.mod": "module",
        "pkg.mod.f": "function",
        "pkg.mod.C": "class",
        "pkg.mod.C.g": "function",
    }


def test_calls_attributed_to_enclosing_function() -> None:
    module = parse_source("def f():\n    g()\n\ndef g():\n    pass\n", "m", "m.py")
    assert ("m.f", "g") in {(c.caller, c.callee) for c in module.calls}


def test_nested_attribute_call_is_dotted() -> None:
    module = parse_source("def f():\n    db.session.commit()\n", "m", "m.py")
    assert ("m.f", "db.session.commit") in {(c.caller, c.callee) for c in module.calls}


def test_relative_import_resolves_to_package() -> None:
    module = parse_source("from .utils import validate as v\n", "pkg.auth", "pkg/auth.py")
    assert module.imports["v"] == "pkg.utils.validate"


def test_inheritance_recorded() -> None:
    module = parse_source("class A(Base):\n    pass\n", "m", "m.py")
    assert (module.inherits[0].subclass, module.inherits[0].base) == ("m.A", "Base")


def test_line_spans_captured_for_citations() -> None:
    module = parse_source("def f():\n    return 1\n", "m", "m.py")
    func = next(n for n in module.nodes if n.qualified_name == "m.f")
    assert (func.start_line, func.end_line) == (1, 2)


def test_docstring_captured() -> None:
    module = parse_source('def f():\n    "doc"\n    return 1\n', "m", "m.py")
    func = next(n for n in module.nodes if n.qualified_name == "m.f")
    assert func.docstring == "doc"


def test_syntax_error_file_is_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def (:\n", encoding="utf-8")
    assert parse_file(bad, tmp_path) is None

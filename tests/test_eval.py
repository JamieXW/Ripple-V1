"""Eval tests: metric math, harness aggregation, and the git miner on a temp repo."""

from __future__ import annotations

from pathlib import Path

import git
from git import Actor

from ripple.eval import EvalExample, build_examples, evaluate, precision_recall
from ripple.graph import build_graph
from ripple.mining import mine_commits
from ripple.parsing import parse_source


def test_precision_recall_basic() -> None:
    precision, recall = precision_recall({"a", "b", "c"}, {"a", "b"})
    assert precision == 2 / 3
    assert recall == 1.0


def test_precision_is_none_without_prediction() -> None:
    precision, recall = precision_recall(set(), {"a"})
    assert precision is None
    assert recall == 0.0


def _two_module_graph() -> object:
    # pkg.m.a -> pkg.m.b ; pkg.n.c -> pkg.m.b  (b is called from both modules)
    m = parse_source("def a():\n    return b()\n\ndef b():\n    return 1\n", "pkg.m", "pkg/m.py")
    n = parse_source("from pkg.m import b\n\ndef c():\n    return b()\n", "pkg.n", "pkg/n.py")
    return build_graph([m, n])


def test_evaluate_scores_perfect_prediction() -> None:
    graph = _two_module_graph()
    example = EvalExample("sha1", "pkg.m.b", frozenset({"pkg/n.py"}))
    report = evaluate(graph, [example])  # type: ignore[arg-type]
    assert report.n_examples == 1
    assert report.n_with_prediction == 1
    assert report.coverage == 1.0
    assert report.micro_recall == 1.0  # pkg/n.py is in the predicted blast radius
    assert report.mean_precision == 1.0  # only co-changed file predicted (own file excluded)


def test_build_examples_skips_unknown_and_self_only() -> None:
    graph = _two_module_graph()
    changes = [
        type(
            "C",
            (),
            {
                "sha": "s",
                "changed_files": frozenset({"pkg/m.py"}),
                "changed_functions": frozenset({"pkg.m.b"}),
            },
        )(),
        type(
            "C",
            (),
            {
                "sha": "s2",
                "changed_files": frozenset({"pkg/x.py"}),
                "changed_functions": frozenset({"pkg.does.not.exist"}),
            },
        )(),
    ]
    examples = build_examples(changes, graph)  # type: ignore[arg-type]
    # First: only co-changed file is b's own file -> dropped. Second: seed unknown -> dropped.
    assert examples == []


def test_mine_commits_maps_changed_lines_to_functions(tmp_path: Path) -> None:
    actor = Actor("Test", "test@example.com")
    repo = git.Repo.init(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (pkg / "b.py").write_text("def g():\n    return 2\n", encoding="utf-8")
    repo.index.add(["pkg/__init__.py", "pkg/a.py", "pkg/b.py"])
    repo.index.commit("init", author=actor, committer=actor)

    (pkg / "a.py").write_text("def f():\n    return 11\n", encoding="utf-8")
    (pkg / "b.py").write_text("def g():\n    return 22\n", encoding="utf-8")
    repo.index.add(["pkg/a.py", "pkg/b.py"])
    repo.index.commit("change f and g", author=actor, committer=actor)

    changes = mine_commits(tmp_path, max_count=10, min_files=2, max_files=10)
    assert len(changes) == 1  # the root commit is skipped (no single parent)
    change = changes[0]
    assert change.changed_files == frozenset({"pkg/a.py", "pkg/b.py"})
    assert {"pkg.a.f", "pkg.b.g"} <= change.changed_functions

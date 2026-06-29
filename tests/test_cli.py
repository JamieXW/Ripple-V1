"""Smoke tests for the CLI skeleton (M0). These lock in that the entry point and
commands wire up correctly before any real logic exists."""

from __future__ import annotations

from typer.testing import CliRunner

from ripple import __version__
from ripple.cli import app

runner = CliRunner()


def test_help_runs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ripple" in result.stdout.lower()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_subcommands_registered() -> None:
    help_text = runner.invoke(app, ["--help"]).stdout
    for command in ("index", "search", "impact", "bench"):
        assert command in help_text

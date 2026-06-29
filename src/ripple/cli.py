"""Ripple command-line interface (Typer).

Commands mirror the four interfaces in the spec. They are stubs for now — each
prints which milestone implements it — so the skeleton is runnable from day one
(``ripple --help``) and we fill them in as we build.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ripple import __version__

app = typer.Typer(
    name="ripple",
    help="Hybrid graph + semantic code-intelligence for Python codebases.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ripple {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Ripple: ask what breaks if you change a piece of code, and where things live."""


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to the Python repository to index."),
) -> None:
    """Full index of a repository (parse -> graph + embeddings -> store)."""
    console.print(f"[yellow]not implemented yet[/] (M1/M3): would index {repo_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="A 'where/how' question."),
    k: int = typer.Option(5, "--k", help="Number of results to return."),
) -> None:
    """Semantic search: ranked code locations with citations."""
    console.print(f"[yellow]not implemented yet[/] (M3): would search {query!r} (k={k})")


@app.command()
def impact(
    symbol: str = typer.Argument(..., help="Qualified symbol, e.g. module.func."),
) -> None:
    """Change-impact: transitive callers/dependents (the blast radius)."""
    console.print(f"[yellow]not implemented yet[/] (M1): would compute blast radius of {symbol}")


@app.command()
def bench() -> None:
    """Run the benchmark suite."""
    console.print("[yellow]not implemented yet[/] (M7): would run benchmarks")


if __name__ == "__main__":
    app()

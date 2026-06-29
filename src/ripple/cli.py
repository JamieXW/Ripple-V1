"""Ripple command-line interface (Typer).

Commands mirror the four interfaces in the spec. They are stubs for now — each
prints which milestone implements it — so the skeleton is runnable from day one
(``ripple --help``) and we fill them in as we build.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ripple import __version__
from ripple.eval import ImpactEvalReport, run_impact_eval
from ripple.graph.models import ImpactResult
from ripple.indexing import index_repo, load_graph, save_graph

app = typer.Typer(
    name="ripple",
    help="Hybrid graph + semantic code-intelligence for Python codebases.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

eval_app = typer.Typer(help="Evaluate Ripple against ground truth.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


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
    repo_path: Path = typer.Argument(
        ...,
        help="Path to the Python repository to index.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Full index of a repository: parse -> build call graph -> persist."""
    repo_path = repo_path.resolve()
    with console.status(f"Indexing {repo_path}…", spinner="dots"):
        graph = index_repo(repo_path)
        dest = save_graph(graph)
    stats = graph.stats
    console.print(f"[green]Indexed[/] {repo_path}")
    console.print(
        f"  nodes: [bold]{graph.graph.number_of_nodes()}[/]   "
        f"edges: [bold]{graph.graph.number_of_edges()}[/]"
    )
    console.print(
        f"  references resolved: [bold]{stats.resolved}/{stats.total}[/] "
        f"({stats.resolution_rate:.0%})"
    )
    console.print(
        f"  [dim]unresolved — external {stats.external}, ambiguous {stats.ambiguous}, "
        f"unknown {stats.unknown}, self-miss {stats.self_miss}[/]"
    )
    console.print(f"  saved: [dim]{dest}[/]")


@app.command()
def search(
    query: str = typer.Argument(..., help="A 'where/how' question."),
    k: int = typer.Option(5, "--k", help="Number of results to return."),
) -> None:
    """Semantic search: ranked code locations with citations."""
    console.print(f"[yellow]not implemented yet[/] (M3): would search {query!r} (k={k})")


@app.command()
def impact(
    symbol: str = typer.Argument(..., help="Symbol to analyze, e.g. module.func or Class.method."),
) -> None:
    """Change-impact: transitive callers/dependents (the blast radius)."""
    try:
        graph = load_graph()
    except FileNotFoundError:
        console.print("[red]No index found.[/] Run [bold]ripple index <repo>[/] first.")
        raise typer.Exit(1) from None

    matches = graph.resolve_symbol(symbol)
    if not matches:
        console.print(f"[red]Symbol not found:[/] {symbol!r}")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous symbol[/] {symbol!r} matches {len(matches)} definitions:")
        for match in matches:
            console.print(f"  • {match}")
        console.print("Re-run with a more qualified name.")
        raise typer.Exit(1)

    _print_impact(graph.impact(matches[0]))


def _print_impact(result: ImpactResult) -> None:
    target = result.target
    console.print(
        f"\n[bold]Impact of[/] {target.qualified_name} "
        f"[dim]({target.file_path}:{target.start_line})[/]"
    )
    if not result.affected:
        console.print("  [dim]Nothing depends on this — no known callers in the index.[/]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("dependency")
    table.add_column("symbol")
    table.add_column("location", style="dim")
    for affected in result.affected:
        table.add_row(
            "direct" if affected.direct else "transitive",
            affected.node.qualified_name,
            f"{affected.node.file_path}:{affected.node.start_line}",
        )
    console.print(table)
    console.print(
        f"[dim]{result.direct_count} direct, {len(result.affected)} total in the blast radius.[/]"
    )


@app.command()
def bench() -> None:
    """Run the benchmark suite."""
    console.print("[yellow]not implemented yet[/] (M7): would run benchmarks")


@eval_app.command("impact")
def eval_impact(
    repo_path: Path = typer.Argument(
        ...,
        help="Git repository to evaluate.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    commits: int = typer.Option(300, "--commits", help="Max commits to scan from HEAD."),
    max_files: int = typer.Option(
        10, "--max-files", help="Skip commits touching more than this many .py files."
    ),
) -> None:
    """Grade impact predictions against git commit history (precision / recall)."""
    repo_path = repo_path.resolve()
    with console.status("Indexing + mining git history…", spinner="dots"):
        report = run_impact_eval(repo_path, max_count=commits, max_files=max_files)
    _print_eval(report, repo_path)


def _print_eval(report: ImpactEvalReport, repo_path: Path) -> None:
    if report.n_examples == 0:
        console.print(
            "[yellow]No gradable examples.[/] No recent commits matched the filters, or no "
            "changed function still exists at HEAD."
        )
        return
    console.print(f"\n[bold]Impact eval[/] on {repo_path.name}")
    console.print(
        f"[dim]{report.n_examples} seed examples from {report.n_commits} commits; "
        f"coverage {report.coverage:.0%} (examples with a non-empty prediction)[/]"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_row(
        "macro (per example)", f"{report.mean_precision:.3f}", f"{report.mean_recall:.3f}"
    )
    table.add_row("micro (pooled)", f"{report.micro_precision:.3f}", f"{report.micro_recall:.3f}")
    console.print(table)
    console.print(
        f"[dim]Recall is coverage-bound: on the {report.coverage:.0%} of seeds where Ripple "
        f"predicts anything, mean recall is {report.mean_recall_when_predicted:.3f}.[/]"
    )
    console.print(
        "[dim]Ground truth = files co-changed in the same commit (a noisy proxy). "
        "Graph built at HEAD; older commits drift.[/]"
    )


if __name__ == "__main__":
    app()

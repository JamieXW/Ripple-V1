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
from sqlalchemy.exc import OperationalError

from ripple import __version__
from ripple.db import (
    init_db,
    load_graph_from_db,
    search_chunks,
    session_scope,
    stored_model_name,
    write_index,
)
from ripple.embeddings import Embedder, build_vector_index
from ripple.eval import ImpactEvalReport, SearchEvalReport, evaluate_search, run_impact_eval
from ripple.graph import build_graph
from ripple.graph.models import ImpactResult
from ripple.parsing import parse_repo
from ripple.parsing.models import CodeNode

app = typer.Typer(
    name="ripple",
    help="Hybrid graph + semantic code-intelligence for Python codebases.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

eval_app = typer.Typer(help="Evaluate Ripple against ground truth.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


def _db_unreachable() -> None:
    console.print(
        "[red]Cannot reach Postgres.[/] Start it with [bold]docker compose up -d[/] "
        "(or set RIPPLE_DATABASE_URL)."
    )


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
    """Full index of a repository: parse -> call graph + embeddings -> Postgres."""
    repo_path = repo_path.resolve()
    with console.status(f"Parsing {repo_path}…", spinner="dots"):
        modules = parse_repo(repo_path)
    with console.status("Building call graph…", spinner="dots"):
        graph = build_graph(modules, repo_root=str(repo_path))
    with console.status("Embedding chunks (downloads the model on first run)…", spinner="dots"):
        vectors = build_vector_index(modules, repo_path, Embedder())
    try:
        init_db()
        with console.status("Writing to Postgres…", spinner="dots"), session_scope() as session:
            write_index(session, graph, vectors)
    except OperationalError:
        _db_unreachable()
        raise typer.Exit(1) from None

    stats = graph.stats
    dim = int(vectors.matrix.shape[1]) if vectors.matrix.shape[0] else 0
    console.print(f"[green]Indexed[/] {repo_path}")
    console.print(
        f"  graph: [bold]{graph.graph.number_of_nodes()}[/] nodes, "
        f"[bold]{graph.graph.number_of_edges()}[/] edges "
        f"([dim]{stats.resolved}/{stats.total} refs resolved, {stats.resolution_rate:.0%}[/])"
    )
    console.print(
        f"  vectors: [bold]{len(vectors.nodes)}[/] chunks embedded "
        f"([dim]dim {dim}, {vectors.model_name}[/])"
    )
    console.print("  stored in [dim]Postgres[/] (HNSW index on chunks.embedding)")


@app.command()
def search(
    query: str = typer.Argument(..., help="A 'where/how' question."),
    k: int = typer.Option(5, "--k", help="Number of results to return."),
) -> None:
    """Semantic search: ranked code locations with citations."""
    try:
        with session_scope() as session:
            model_name = stored_model_name(session)
    except OperationalError:
        _db_unreachable()
        raise typer.Exit(1) from None
    if model_name is None:
        console.print("[red]No index found.[/] Run [bold]ripple index <repo>[/] first.")
        raise typer.Exit(1)

    with console.status("Embedding query…", spinner="dots"):
        query_vector = Embedder(model_name).embed_query(query)
    with session_scope() as session:
        results = search_chunks(session, query_vector, k=k)
    _print_search(query, results)


def _print_search(query: str, results: list[tuple[CodeNode, float]]) -> None:
    console.print(f"\n[bold]Search[/] {query!r}")
    if not results:
        console.print("  [dim]No results — is anything indexed?[/]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("score", justify="right")
    table.add_column("symbol")
    table.add_column("location", style="dim")
    for node, score in results:
        table.add_row(
            f"{score:.3f}",
            node.qualified_name,
            f"{node.file_path}:{node.start_line}",
        )
    console.print(table)


@app.command()
def impact(
    symbol: str = typer.Argument(..., help="Symbol to analyze, e.g. module.func or Class.method."),
) -> None:
    """Change-impact: transitive callers/dependents (the blast radius)."""
    try:
        with session_scope() as session:
            graph = load_graph_from_db(session)
    except OperationalError:
        _db_unreachable()
        raise typer.Exit(1) from None
    if not graph.nodes:
        console.print("[red]No index found.[/] Run [bold]ripple index <repo>[/] first.")
        raise typer.Exit(1)

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


@eval_app.command("search")
def eval_search(
    repo_path: Path = typer.Argument(
        ...,
        help="Repository to evaluate semantic search on.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Grade semantic search on held-out docstring queries (recall@k, MRR, nDCG)."""
    repo_path = repo_path.resolve()
    with console.status(f"Parsing {repo_path}…", spinner="dots"):
        modules = parse_repo(repo_path)
    with console.status("Embedding corpus + queries (model downloads on first run)…"):
        report = evaluate_search(modules, repo_path, Embedder())
    _print_search_eval(report, repo_path)


def _print_search_eval(report: SearchEvalReport, repo_path: Path) -> None:
    if report.n_queries == 0:
        console.print("[yellow]No held-out docstring queries found[/] — nothing to grade.")
        return
    console.print(f"\n[bold]Semantic search eval[/] on {repo_path.name}")
    console.print(
        f"[dim]{report.n_corpus} corpus chunks (docstrings stripped), "
        f"{report.n_queries} held-out queries, {report.n_train_pairs} train pairs "
        f"reserved for fine-tuning[/]"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("recall@1", f"{report.recall_at_1:.3f}")
    table.add_row("recall@5", f"{report.recall_at_5:.3f}")
    table.add_row("recall@10", f"{report.recall_at_10:.3f}")
    table.add_row("MRR", f"{report.mrr:.3f}")
    table.add_row("nDCG@10", f"{report.ndcg_at_10:.3f}")
    console.print(table)
    console.print(
        "[dim]Query = held-out docstring; correct answer = its own function (corpus has "
        "docstrings stripped to prevent leakage). This is the Tier-0 baseline the "
        "fine-tuned reranker (M5b) must beat.[/]"
    )


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

# Ripple

A hybrid **graph + semantic** code-intelligence engine for Python codebases.

Most AI code tools answer questions by embedding similarity — great for *"where is auth
handled?"*, structurally weak for *"what breaks if I change this function?"*. Ripple fuses
a real call/import/inherit **graph** (answers change-impact via traversal) with **semantic
search** (answers where/how), and validates its blast-radius predictions against real git
history. Every answer carries file:line citations.

> Status: **M1 — call graph + impact.** `index` and `impact` are live; semantic
> search, the ML reranker, and the service layer are upcoming milestones.

## Quickstart (dev)

```bash
uv sync --extra dev      # create .venv and install (incl. dev tools)
uv run ripple --help     # the CLI skeleton
uv run pytest            # smoke tests
```

## Commands

| Command | Question it answers | Status |
|---|---|---|
| `ripple index <repo>` | — (build the call graph) | ✅ M1 (graph; embeddings in M3) |
| `ripple impact <symbol>` | what breaks if I change X? | ✅ M1 |
| `ripple search "<q>"` | where/how is X handled? | M3 |
| `ripple bench` | — (benchmark suite) | M7 |

```console
$ ripple index path/to/repo
$ ripple impact flask.views.View      # who breaks if View changes?
```

### Known limitations (by design, measured in M2)

Call resolution is static, so dynamic dispatch, inherited-method calls, and methods
invoked on local variables aren't linked — we resolve what's statically certain and
**count** the rest rather than guess. The eval harness (M2) reports impact precision /
recall against real git history.

## Architecture

```
repo -> parse(AST) -> { call graph, embeddings } -> Postgres + pgvector
                                                          |
query -> retrieval (semantic seed -> graph expand -> rerank) -> answer + citations

build-time: mine pairs -> fine-tune reranker;  evaluate vs git history
```

The full architecture diagram, benchmark table, and honest limitations will live here as
the build progresses.

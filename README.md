# Ripple

A hybrid **graph + semantic** code-intelligence engine for Python codebases.

Most AI code tools answer questions by embedding similarity — great for *"where is auth
handled?"*, structurally weak for *"what breaks if I change this function?"*. Ripple fuses
a real call/import/inherit **graph** (answers change-impact via traversal) with **semantic
search** (answers where/how), and validates its blast-radius predictions against real git
history. Every answer carries file:line citations.

> Status: **early build (M0 — scaffold).** See the milestone plan for what's live.

## Quickstart (dev)

```bash
uv sync --extra dev      # create .venv and install (incl. dev tools)
uv run ripple --help     # the CLI skeleton
uv run pytest            # smoke tests
```

## Commands

| Command | Question it answers | Status |
|---|---|---|
| `ripple index <repo>` | — (build the indexes) | M1 / M3 |
| `ripple search "<q>"` | where/how is X handled? | M3 |
| `ripple impact <symbol>` | what breaks if I change X? | M1 |
| `ripple bench` | — (benchmark suite) | M7 |

## Architecture

```
repo -> parse(AST) -> { call graph, embeddings } -> Postgres + pgvector
                                                          |
query -> retrieval (semantic seed -> graph expand -> rerank) -> answer + citations

build-time: mine pairs -> fine-tune reranker;  evaluate vs git history
```

The full architecture diagram, benchmark table, and honest limitations will live here as
the build progresses.

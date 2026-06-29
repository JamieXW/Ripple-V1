# Ripple

A hybrid **graph + semantic** code-intelligence engine for Python codebases.

Most AI code tools answer questions by embedding similarity — great for *"where is auth
handled?"*, structurally weak for *"what breaks if I change this function?"*. Ripple fuses
a real call/import/inherit **graph** (answers change-impact via traversal) with **semantic
search** (answers where/how), and validates its blast-radius predictions against real git
history. Every answer carries file:line citations.

> Status: **M2 — eval harness.** `index`, `impact`, and `eval impact` (graded against
> git history) are live; semantic search, the ML reranker, and the service layer are
> upcoming milestones.

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
| `ripple eval impact <repo>` | how accurate are impact predictions? | ✅ M2 |
| `ripple search "<q>"` | where/how is X handled? | M3 |
| `ripple bench` | — (benchmark suite) | M7 |

```console
$ ripple index path/to/repo
$ ripple impact flask.views.View      # who breaks if View changes?
$ ripple eval impact path/to/repo     # grade predictions vs. git history
```

### Evaluation: impact vs. git history (M2 baseline)

Ripple grades its own blast-radius predictions against ground truth mined for free from
git: the files a commit *actually* co-changed. On a sample of Flask commits (file-level):

| metric | value | reading |
|---|---|---|
| precision | **~0.28** | of predicted files, share that really co-changed |
| recall (overall) | **~0.01** | bounded by coverage (below) |
| recall (when predicting) | **~0.15** | on the seeds it can predict |
| coverage | **~8%** | seeds with any statically-resolved caller |

The story is **coverage**: `recall ≈ coverage × recall-when-predicting`. Static call
resolution misses dynamic dispatch, inherited-method calls, and methods invoked on local
variables, so most changed functions have no resolved caller — that's the ceiling on
recall, and the lever later milestones pull. Co-change is also a noisy proxy for causal
impact. These limits are **measured and reported**, not hidden — this honest baseline is
the number future work is graded against.

## Architecture

```
repo -> parse(AST) -> { call graph, embeddings } -> Postgres + pgvector
                                                          |
query -> retrieval (semantic seed -> graph expand -> rerank) -> answer + citations

build-time: mine pairs -> fine-tune reranker;  evaluate vs git history
```

The full architecture diagram, benchmark table, and honest limitations will live here as
the build progresses.

# Ripple

A hybrid **graph + semantic** code-intelligence engine for Python codebases.

Most AI code tools answer questions by embedding similarity — great for *"where is auth
handled?"*, structurally weak for *"what breaks if I change this function?"*. Ripple fuses
a real call/import/inherit **graph** (answers change-impact via traversal) with **semantic
search** (answers where/how), and validates its blast-radius predictions against real git
history. Every answer carries file:line citations.

> Status: **M5a — semantic eval baseline.** Search accuracy is now *measured* on held-out
> docstring queries (leakage-safe). `index`, `impact`, `search`, `eval impact`, and
> `eval search` are live; the fine-tuned reranker (M5b) and service layer are upcoming.

## Quickstart (dev)

```bash
uv sync --extra dev      # create .venv and install (incl. dev tools)
docker compose up -d     # start Postgres + pgvector (host port 55432)
uv run ripple index path/to/repo   # parse -> graph + embeddings -> Postgres
uv run pytest            # DB tests skip automatically if Postgres is down
```

The database connection defaults to `postgresql+psycopg://ripple:ripple@localhost:55432/ripple`
(override with `RIPPLE_DATABASE_URL`). Host port 55432 avoids clashing with other local
Postgres instances.

## Commands

| Command | Question it answers | Status |
|---|---|---|
| `ripple index <repo>` | — (build graph + embeddings) | ✅ M1 / M3 |
| `ripple impact <symbol>` | what breaks if I change X? | ✅ M1 |
| `ripple search "<q>"` | where/how is X handled? | ✅ M3 |
| `ripple eval impact <repo>` | how accurate are impact predictions? | ✅ M2 |
| `ripple eval search <repo>` | how accurate is semantic search? | ✅ M5a |
| `ripple bench` | — (benchmark suite) | M7 |

```console
$ ripple index path/to/repo
$ ripple impact flask.views.View                       # who breaks if View changes?
$ ripple search "serialize an object to a JSON response"  # where/how is it handled?
$ ripple eval impact path/to/repo                      # grade predictions vs. git history
```

### Semantic search baseline (M5a)

Search is graded on **held-out docstring queries**: a function's docstring becomes the
query, the function itself is the single correct answer, and the corpus has all docstrings
**stripped** so a query can never match its own text (leakage). Tier-0 baseline on Flask
(1,618 chunks, 56 held-out queries; 245 train pairs reserved for fine-tuning):

| metric | Tier-0 (all-MiniLM-L6-v2) | + fine-tuned reranker (M5b) |
|---|---|---|
| recall@1 | 0.429 | — |
| recall@5 | 0.607 | — |
| recall@10 | 0.714 | — |
| MRR | 0.502 | — |
| nDCG@10 | 0.558 | — |

### Semantic search (M3)

AST-aware chunking (one chunk per function/class — reusing the M1 parse, not fixed line
windows) → a Tier-0 embedding model (`all-MiniLM-L6-v2`) → brute-force cosine nearest-
neighbor. On Flask, *"serialize an object to a JSON response"* surfaces `flask.json.tag`'s
`to_json` methods; *"load the session from a signed cookie"* surfaces
`SecureCookieSessionInterface`. Modest scores (~0.5–0.6) are expected from a general model —
the M5 fine-tune is where that lifts. Vectors are in-memory NumPy for now; pgvector ANN
arrives in M4.

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

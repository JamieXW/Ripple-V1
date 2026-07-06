# Ripple

[![CI](https://github.com/JamieXW/Ripple-V1/actions/workflows/ci.yml/badge.svg)](https://github.com/JamieXW/Ripple-V1/actions/workflows/ci.yml)

A hybrid **graph + semantic** code-intelligence engine for Python codebases.

Most AI code tools answer questions by embedding similarity — great for *"where is auth
handled?"*, structurally weak for *"what breaks if I change this function?"*. Ripple fuses
a real call/import/inherit **graph** (answers change-impact via traversal) with **semantic
search** (answers where/how), and validates its blast-radius predictions against real git
history. Every answer carries file:line citations.

> Status: **M8-prep — deploy-ready.** Production Docker image (CPU-only torch, 3.5GB vs 17.5GB with CUDA),
> full-stack compose profile, a web demo UI at `/` showing live per-stage traces, and a
> [deploy runbook](docs/DEPLOY.md) — shipping is one click when application season needs
> the link.

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
| `ripple eval search <repo> [--reranker …]` | how accurate is semantic search? | ✅ M5a/M5b |
| `ripple train reranker <repo>` | — (fine-tune the reranker on mined pairs) | ✅ M5b |
| `ripple serve` | — (run the API service) | ✅ M6 |
| `ripple mcp` | — (run the MCP server for AI agents) | ✅ M6.5 |
| `ripple bench` | — (benchmark suite) | M7 |

```console
$ ripple index path/to/repo
$ ripple impact flask.views.View                       # who breaks if View changes?
$ ripple search "serialize an object to a JSON response"  # where/how is it handled?
$ ripple eval impact path/to/repo                      # grade predictions vs. git history
```

## Benchmarks (M7 — measured on an Apple-Silicon laptop)

### Incremental reindexing (warm service, single-file change)

| repo | scale | full index | incremental | speedup |
|---|---|---|---|---|
| Flask | 83 files · 1,620 chunks | 8.8s | **0.56s** | **15.7×** |
| Django | 2,924 files · 43,431 chunks · 46k nodes | 221s | **11.1s** | **20×** |

Change detection = per-file sha256 (`file_hashes` table). Only changed files' chunks
re-embed (embeddings depend solely on their own text); parse + graph rebuild in full —
cheap and staleness-proof. The Django breakdown shows the *next* levers honestly:
at 43k chunks the incremental cost is parse (4.9s) + graph rewrite (4.7s db), embed
is only 0.6s — per-file parse caching and graph-delta writes are documented future work.

### Serving latency & throughput (`ripple bench`, Flask index, 60 reqs @ concurrency 8)

| scenario | p50 | p95 | p99 | req/s |
|---|---|---|---|---|
| `/impact` (graph traversal) | 4ms | 10ms | 14ms | **1,463** |
| `/search` no rerank | 52ms | 68ms | 76ms | **151** |
| `/search` cache hit | 5ms | — | — | — |
| `/search` full pipeline (rerank) | 1,259ms | 1,830ms | 1,860ms | 6.0 |

**Where it breaks:** sweeping concurrency 1→16 on the full pipeline, throughput
saturates at **~6 req/s** while p50 grows linearly (210ms → 2,598ms) — classic queueing
on a serialized resource: cross-encoder inference is the ceiling. Mitigations measured
(no-rerank path, caching) and planned (distill the reranker into the bi-encoder).

### Trace-driven optimizations

- Query embedding **499ms → 3.5ms**: single-query inference pinned to CPU — at batch
  size 1, GPU launch overhead dominates (measured: cpu 3ms vs mps 6ms; bulk indexing
  keeps the GPU, where it wins 2×).
- Rerank inputs truncated 512 → 256 tokens: **~3× faster and better ranking** — held-out
  recall@1 0.357 → 0.411, MRR 0.500 → 0.527 (noisy long-function tails were hurting).
- At 43k vectors, pgvector retrieve grew 19ms → ~200ms; `EXPLAIN ANALYZE` confirms the
  HNSW index is used (~93ms scan, default `ef_search`) — tuning documented as next work.

## MCP server — Ripple as agent tooling (M6.5)

```bash
uv run ripple mcp        # stdio MCP server (spawned automatically by clients)
```

The committed [`.mcp.json`](.mcp.json) makes Claude Code pick Ripple up in this project
automatically. Tools exposed (all results carry file:line citations):

| Tool | The question an agent asks |
|---|---|
| `impact_of(symbol)` | *"What breaks if I change this?"* — call **before editing** |
| `search_code(query, k)` | *"Where is X handled?"* — hybrid semantic + graph retrieval |
| `index_status()` | *"Is anything indexed?"* |

Runs in-process against the same pipeline as the API (the HTTP service doesn't need to
be running). Verified over a real stdio session against the Flask index: an agent asking
`impact_of("flask.views.View")` gets the 25-symbol blast radius with citations before
touching the code.

## API service (M6)

```bash
docker compose up -d                       # Postgres+pgvector, Redis
RIPPLE_RERANKER_MODEL=models/reranker uv run ripple serve   # http://127.0.0.1:8000/docs
```

| Endpoint | What it does |
|---|---|
| `GET /search?q=…&k=5&rerank=true&expand=true` | Hybrid retrieval: pgvector seeds → 1-hop graph expansion → cross-encoder rerank |
| `GET /impact?symbol=…` | Blast radius (direct + transitive) with citations |
| `POST /index {"repo_path": …}` | Background (re)index; progress via `/health` |
| `GET /health` | Index status, **cache hit rate**, model names |

Every `/search` response carries `meta.timings_ms` — OpenTelemetry-backed per-stage
timings. A warm Flask query: `embed 499ms · retrieve 19ms · expand 3ms · rerank 537ms`,
i.e. **98% of latency is model inference, 2% is the database** — which is exactly the
kind of thing tracing is for (and what M7 optimizes). Responses are cached in Redis
(TTL, flushed on reindex); async FastAPI with the async SQLAlchemy driver for I/O and
`asyncio.to_thread` for CPU-bound inference. Degrades gracefully: no Redis → caching
off; no Postgres → boots empty and says so.

### Semantic search baseline (M5a)

Search is graded on **held-out docstring queries**: a function's docstring becomes the
query, the function itself is the single correct answer, and the corpus has all docstrings
**stripped** so a query can never match its own text (leakage). Tier-0 baseline on Flask
(1,618 chunks, 56 held-out queries; 245 train pairs reserved for fine-tuning):

| metric | bi-encoder only | + zero-shot reranker | + fine-tuned reranker (M5b) |
|---|---|---|---|
| recall@1 | **0.429** | 0.143 | 0.357 |
| recall@5 | 0.607 | 0.607 | **0.679** |
| recall@10 | 0.714 | 0.661 | **0.732** |
| MRR | **0.502** | 0.296 | 0.500 |
| nDCG@10 | 0.558 | 0.391 | **0.574** |

Three honest findings: (1) an off-the-shelf English reranker is *worse than useless* on
code — recall@1 collapses 0.43→0.14 (the domain gap, measured); (2) fine-tuning on
~1.2k pairs mined from docstrings + retriever hard negatives recovers most of it —
recall@1 2.5× vs zero-shot, MRR 0.30→0.50 — a clean isolation of what the mined data
contributes; (3) the tuned pipeline beats the bi-encoder on recall@5/@10 and nDCG but
not yet at rank-1 — the reranker is data-starved at 245 positives from one repo.
Next levers: mine more repos, add commit-message pairs, score fusion. We deliberately
do not iterate against the held-out set to chase a prettier number.

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

## Web demo & deployment (M8)

The API serves a minimal demo UI at `/` — search and impact side by side, every result
with `file:line` citations, and the **live per-stage timing trace** rendered under each
query (watch embed → retrieve → expand → rerank happen). Run the whole stack in
containers:

```bash
docker compose --profile app up --build      # app on http://localhost:58000
```

The [Dockerfile](Dockerfile) installs **CPU-only torch on Linux** (pinned via uv index
sources) — justified by M7's measurement that serving inference runs faster on CPU at
serving batch sizes — cutting the image from a measured 17.5GB with CUDA dependencies to 3.5GB, with
the embedder pre-baked so first boot doesn't download models. Cloud deployment is a
one-command runbook away: [docs/DEPLOY.md](docs/DEPLOY.md).

## Architecture

```mermaid
flowchart LR
  subgraph index [Indexing — incremental via file hashes]
    R[Python repo] --> P[AST parse]
    P --> G[call graph]
    P --> E[embed changed chunks]
    G --> PG[(Postgres + pgvector)]
    E --> PG
  end
  subgraph query [Query — traced per stage]
    Q[query] --> S[semantic seeds: pgvector HNSW]
    S --> X[graph expansion: 1 hop]
    X --> RR[fine-tuned cross-encoder rerank]
    RR --> A[answers + file:line citations]
  end
  PG --> S
  G -. in-memory graph .-> X
  C[(Redis cache)] --- query
  subgraph ml [Build-time ML]
    M[mine pairs: docstrings + hard negatives] --> T[fine-tune reranker]
    H[git history] --> EV[eval harness: held-out splits]
  end
  T -.-> RR
  AG[AI agents via MCP] --> Q
  U[web UI + REST] --> Q
```

## Honest limitations

- **Static resolution is incomplete:** dynamic dispatch, inherited-method calls, and
  methods on local variables aren't linked — impact recall is coverage-bound (~8% of
  changed functions have a resolved caller; measured, improvement plan documented).
- **Co-change ground truth is noisy** (commits bundle unrelated edits); the graph is
  built at HEAD, so older commits drift.
- **The reranker is data-starved** (245 positives from one repo) — beats the bi-encoder
  on recall@5/@10/nDCG, near-ties rank-1; multi-repo mining is the planned fix.
- **Full-pipeline throughput ceils at ~6 req/s per instance** (serialized cross-encoder
  inference, found by concurrency sweep) — measured mitigations: no-rerank path
  (151 req/s), cache hits (5ms); distillation planned.
- **Python only; one active index** per database.

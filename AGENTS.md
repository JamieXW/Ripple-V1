# Code Intelligence Engine — Project Context

> Handoff + standing context doc for Codex. Read this first, in full, before
> scaffolding or writing code. It captures the design, rationale, decisions, stack,
> structure, data model, interfaces, timeline, and conventions worked out in planning.
> As decisions get made during the build, append them to the relevant section so this
> file stays the source of truth.

---

## 1. Owner goals (why this project exists)

Personal flagship project to stand out for **Summer 2027 internships**. Targets, in
priority order:
1. **Big tech SWE** (primary — optimize the design and the talking points for this)
2. **AI engineer / LLM engineer**
3. **Fintech SWE** (domain-agnostic here, but the rigor maps over)

Endgame: starts as a portfolio piece + live demo, architected cleanly enough that it
*could* take real users later. ~15 hrs/week available. Want it credible before the
**July–Aug application rush**.

Design principle: **the hard part is the systems engineering**, with real ML on top.
That combination is what serves all three targets at once. When trading off, favor
systems depth + measured rigor over feature breadth.

---

## 2. The problem

In a large codebase, two developer questions are painful:
1. **"Where/how" questions** — e.g. *"where is auth handled?"* → a **similarity** problem.
2. **"Impact" questions** — e.g. *"what breaks if I change `validate_token()`?"* → a
   **connection** problem (transitive callers/dependents), NOT similarity.

Question 2 is the high-value, time-wasting one and is where existing tools are weakest.

---

## 3. Why existing tools fall short (our wedge)

Tools like Cursor answer via **embedding similarity**: chunk code → embed each chunk →
nearest-neighbor search vs. the query embedding. Great for "where is X" (similarity), but
structurally weak for "what breaks if I change X" (that's graph traversal). They also
degrade in reliability on very large codebases.

**Our edge:** fuse a real code-dependency **graph** (answers impact via traversal) with
**semantic search** (answers where/how). Combining the two is what incumbents don't do.
We are NOT trying to beat Cursor as a product — we are demonstrating the retrieval +
systems engineering that the teams building these tools do.

---

## 4. Solution architecture

### Phase 1 — Indexing (when pointed at a repo)
1. **Parse** Python files into structure (functions, classes, imports, call relationships)
   using Python's stdlib `ast` (more accurate call resolution for Python than generic
   parsers). Resolve names where feasible; `jedi` is an option for harder resolution.
2. Build **two indexes from the same parse**:
   - **Graph ("wire map")**: nodes = functions/classes/modules; edges = calls / imports /
     inherits. Answers impact questions via traversal.
   - **Semantic index**: chunk code **by function/class** (AST-aware — NOT fixed line
     windows), embed each chunk, store vectors. Answers where/how questions.

### Phase 2 — Query (live service)
- **Where/how**: embed question → nearest-neighbor → return top chunks (+ citations).
- **Impact**: resolve target to a graph node → traverse transitive callers/dependents →
  return affected functions, locations, and which have test coverage.
- **Hybrid**: semantic search finds the seed → graph expands the blast radius → reranker
  orders results. (Optional later: LLM composes a plain-English answer with citations.)

**Every answer returns file + line citations.** This is non-negotiable for credibility.

---

## 5. The two ideas that make this non-generic (protect these)
1. **The graph teaches the ML.** Training a retrieval model needs labeled
   (query, relevant-code) pairs. The call graph supplies them for free: connected
   functions = relevant pairs; distant random pairs = hard negatives.
2. **Git history = ground truth for impact.** A historical commit that changed function X
   also touched some set of files — that set is the *real* blast radius. Evaluate impact
   predictions against what commits actually changed. This rigor is the standout.

---

## 6. ML approach (tiers — start low, climb if time allows)
- **Tier 0 (baseline, no training):** off-the-shelf embedding model as-is = the "before"
  number to beat. Start with `sentence-transformers`; a code-capable model
  (CodeBERT / GraphCodeBERT / CodeT5 family) is ideal, or `all-MiniLM-L6-v2` /
  `bge-small` / `e5` as a fast general baseline. All from Hugging Face.
- **Tier 1 (sweet spot — do first):** fine-tune a small **cross-encoder reranker** on
  mined pairs. Trains in minutes–hours on free Colab. Gets an end-to-end system fast.
- **Tier 2 (deeper, more impressive):** fine-tune the **embedding model itself** on the
  same mined pairs. Do in ~week 3 only if Tier 1 is solid.

**Training data — mined for free, no manual labeling:**
1. **Call graph edges** → relevant pairs; random distant pairs → hard negatives.
2. **Docstrings** → (docstring, function body) = (query, relevant-code) pairs.
3. **Git history / PR titles** → (commit message, changed files) pairs (also doubles as
   impact ground truth). Extract via `git log` (e.g. `GitPython`).

The data-generation pipeline is itself a strong talking point. **The measured lift is the
deliverable** — e.g. precision@5 0.61 → 0.79 on held-out data — not the act of training.

---

## 7. Evaluation (half the signal — build the harness early)
- **Semantic queries:** labeled query → correct-location set. Bootstrap from docstrings
  and real GitHub issues/PRs (a PR "fix auth redirect" → files it changed = ground truth).
  Metrics: **precision@k, recall@k, nDCG@k**.
- **Impact queries:** predicted-affected-files vs. actually-changed-files from git history.
  Metrics: **precision / recall** of the predicted blast radius.
- Always report **before/after** (off-the-shelf vs. fine-tuned) on a held-out split.
- Report failure cases honestly (missed dynamic-dispatch edges, etc.) — maturity signal.

Metric definitions (for the harness):
- *precision@k*: of the top-k returned, fraction that are relevant.
- *recall@k*: of all relevant items, fraction appearing in the top-k.
- *nDCG@k*: ranking quality, rewarding relevant hits nearer the top.

---

## 8. Systems layer (PRIMARY — big-tech signal)
- Async **FastAPI** service; **Redis** caching (report cache hit rate); structured
  per-stage request tracing (parse → retrieve → graph-walk → rerank), so you can show
  exactly where latency goes.
- **Incremental reindexing:** on file change, re-parse/re-embed/re-graph only changed
  files **and the functions whose graph-dependents are affected** (minimal correct
  update). Detect changes via per-file content hashes (Merkle-tree approach). Measure
  full-index vs. incremental-update time — make this a **headline**, not a footnote.
- **Benchmarks:** index time; query latency **p50/p95/p99** under concurrent load
  (Locust or k6); throughput; memory footprint. **Load-test until it breaks**, report
  where/why and the fix.
- Index a **genuinely large real repo** for the demo so numbers are credible.

---

## 9. Tech stack / dependencies
- **Language/runtime:** Python 3.11+.
- **Parsing:** stdlib `ast` (primary); `jedi` (optional, harder name resolution).
- **Graph + storage:** **PostgreSQL** for graph (edge tables) AND vectors via the
  **`pgvector`** extension — one store for both keeps the systems story clean.
  `NetworkX` acceptable for a fast in-memory first pass.
- **DB access:** `SQLAlchemy` + `psycopg` (or `asyncpg` for the async service).
- **Embeddings/ML:** `sentence-transformers`, `torch`, `transformers`; fine-tune on Colab.
- **Lexical retrieval (hybrid):** BM25 via `rank_bm25` (or Postgres full-text) to pair
  with dense retrieval.
- **Service:** `FastAPI` + `uvicorn`; `redis` for caching.
- **Git mining:** `GitPython`.
- **CLI:** `Typer`.
- **Load testing:** `locust` or `k6`.
- **Quality:** `pytest`, `ruff`, `black`, `mypy`.
- **Deploy:** Docker; Fly.io or Railway for the public demo.

---

## 10. Proposed repository structure
```
code-intel/
├── AGENTS.md                  # this file
├── README.md                  # architecture diagram, benchmarks, tradeoffs (the writeup)
├── pyproject.toml             # deps, ruff/black/mypy config
├── docker-compose.yml         # postgres+pgvector, redis, app for local dev
├── Dockerfile
├── src/code_intel/
│   ├── config.py              # pydantic-settings; env-driven config
│   ├── parsing/               # ast walk → functions/classes/imports/calls
│   ├── graph/                 # build + store + traverse the dependency graph
│   ├── embeddings/            # chunking, embedding, ANN search (pgvector)
│   ├── retrieval/             # hybrid pipeline: semantic seed + graph expand + rerank
│   ├── indexing/              # full + incremental (hash-based) indexing orchestration
│   ├── mining/                # training-pair miners (docstrings, graph, git history)
│   ├── eval/                  # eval harness: semantic + impact metrics
│   ├── api/                   # FastAPI app, routes, tracing middleware
│   └── cli.py                 # Typer entrypoint
├── scripts/                   # one-off: pull target repo, run benchmarks, train on Colab
├── benchmarks/                # locust/k6 files + recorded results
└── tests/
```

---

## 11. Interfaces
**CLI (Typer):**
- `code-intel index <repo_path>` — full index.
- `code-intel search "<query>"` — where/how; prints ranked locations + citations.
- `code-intel impact <symbol>` — blast radius + test coverage.
- `code-intel bench` — run the benchmark suite.

**HTTP API (FastAPI):**
- `POST /index` — body `{ "repo_path" | "repo_url" }`; kicks off indexing.
- `GET  /search?q=...&k=...` — semantic results with citations.
- `GET  /impact?symbol=...` — affected nodes, locations, test-coverage flags.
- `GET  /health` — liveness + index status.
All responses include file + line citations.

---

## 12. Data model (Postgres + pgvector) — sketch, refine in code
- `nodes(id, repo, kind, qualified_name, file_path, start_line, end_line, docstring)`
  — kind ∈ {function, class, module}.
- `edges(src_id, dst_id, edge_type)` — edge_type ∈ {calls, imports, inherits}.
- `chunks(id, node_id, content, embedding vector(N))` — N = model dim; ANN index on
  `embedding` (pgvector ivfflat/hnsw).
- `coverage(test_node_id, covered_node_id)` — test → code mapping for impact answers.
- `file_hashes(file_path, content_hash, indexed_at)` — drives incremental reindexing.

---

## 13. Decisions (locked defaults — revisit only with reason)
1. **Graph storage:** Postgres edge tables (+ pgvector for vectors). NetworkX only for a
   first in-memory pass; Neo4j only if a concrete need appears.
2. **ML depth:** reranker-first (Tier 1) for end-to-end, then embeddings (Tier 2) ~week 3.
3. **v1 output:** ranked code locations with citations. LLM-written NL answers = later
   thin layer, not v1.
4. **Interface:** CLI first; simple web UI for the demo link. GitHub bot = stretch (best
   "real usage" story, most plumbing).
5. **Target repo for headline benchmark:** **OPEN** — Django / pandas / CPython stdlib, or
   a repo the owner knows well. Decide before wiring the ingestion path.

---

## 14. Honest scope boundaries
- **One language: Python.** Depth in one beats a shaky polyglot demo.
- **Static resolution is imperfect** (monkeypatching, `getattr`, dynamic dispatch → some
  missed edges). **Measure and document** this; don't hide it.
- **Don't add everything.** Breadth that half-works reads worse than a focused build with
  real numbers. Only add systems depth you can instrument and defend.

---

## 15. Headroom (LATER — keep out of v1)
Distributed/parallel indexing across worker processes (top-tier "distributed systems"
signal), a learned ranking model, multi-language support, an agentic layer that chains
queries. Stretch goals only — do not let these creep into v1.

---

## 16. Six-week plan (~15 hrs/week)
- **Week 1 — Foundations + thin slice.** Scaffold repo; write a one-page spec naming exact
  target metrics and the 3–4 résumé bullets up front; pick the target repo; AST parse →
  function/class chunking → Tier-0 baseline embeddings → naive dense retrieval → return a
  result with citation. Crude is fine; get the pipe connected end-to-end.
- **Week 2 — Graph.** Build call/import/inherit graph from the AST; store in Postgres edge
  tables; implement transitive caller/dependent traversal; CLI for both query types.
- **Week 3 — ML depth + eval.** Mine training pairs (docstrings, graph edges/hard
  negatives, git history); fine-tune the reranker (Tier 1); build the eval harness
  (semantic + impact); report before/after. Tier-2 embeddings if time allows.
- **Week 4 — Systems spine.** Async FastAPI service; Redis caching; full hybrid pipeline
  (semantic seed + graph expansion + rerank); structured per-stage tracing.
- **Week 5 — Scale + performance.** Incremental hash-based reindexing; load-test (Locust/k6)
  to breaking point and tune; capture p50/p95/p99, throughput, cache hit rate, memory;
  index a large real repo for credible numbers.
- **Week 6 — Ship + writeup.** Dockerize and deploy (Fly.io/Railway); web UI demo; README
  with architecture diagram, benchmark numbers, tradeoffs, and honest limitations.

---

## 17. Engineering conventions
- `src/` layout; full type hints; `ruff` + `black` + `mypy` clean.
- Config via `pydantic-settings`, env-driven (no hardcoded paths/secrets).
- Structured logging from day one (it becomes the observability/tracing layer).
- `pytest` for the parser, graph traversal, and eval metrics especially.
- Pin dependencies. Commit small and often (the git history is also project narrative).
- Keep the README benchmark table updated as numbers change — it's a primary deliverable.

---

## 18. Target résumé bullets (what success looks like)
- *Systems:* "Built an async code-intelligence service with incremental hash-based
  reindexing, sustaining ~X req/s at p99 < Y ms over a K-LOC Python repo."
- *ML:* "Fine-tuned a code retrieval model on call-graph-mined pairs, lifting
  impact-prediction recall A→B, validated against git commit history."
- *Judgment:* "Designed a hybrid graph + semantic retriever that answers change-impact
  queries pure-embedding tools handle poorly."

---

## 19. Mini-glossary
- **Embedding:** a vector of numbers encoding the meaning of text/code; similar meaning →
  closer vectors.
- **Embedding model:** the (pretrained, downloaded) model that produces those vectors.
- **Reranker (cross-encoder):** a small model that scores (query, candidate) relevance to
  reorder retrieval results.
- **Hard negatives:** plausible-but-wrong examples that make training effective.
- **ANN / nearest-neighbor:** fast "find the closest vectors" search (here via pgvector).
- **Call graph / dependency graph:** nodes = code units, edges = calls/imports/inherits.
- **Blast radius:** everything transitively affected by changing a given node.
- **Incremental reindexing:** re-processing only changed files + affected dependents,
  using content hashes to detect change.
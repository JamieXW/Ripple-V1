# Deploying Ripple (runbook — prepared in M8, executed when ready)

The image is fully self-contained (CPU-only torch, embedder pre-baked), so deploying
is: run the container + give it Postgres(pgvector) + Redis + two env vars.

## Verify locally first (no cloud required)

```bash
docker compose --profile app up --build     # app on http://localhost:58000
# then index the demo repo through the container:
git clone --depth 1 https://github.com/pallets/flask /tmp/flask   # any path works
curl -X POST localhost:58000/index -H 'Content-Type: application/json' \
     -d '{"repo_path": "/tmp/flask"}'   # note: path must be visible to the container
```

For the compose profile, the simplest demo-repo approach is to clone *inside* the
container: `docker compose exec app sh -c "apt-get update && apt-get install -y git &&
git clone --depth 1 https://github.com/pallets/flask /tmp/flask"` then POST /index.

## Railway (recommended: simplest managed pgvector + Redis)

1. `railway init` in the repo (or "New Project → Deploy from GitHub" in the dashboard).
   Railway detects the Dockerfile automatically.
2. Add **PostgreSQL** (Railway's image ships pgvector; `init_db` runs
   `CREATE EXTENSION IF NOT EXISTS vector` itself) and **Redis** from the plugin menu.
3. Set service variables:
   - `RIPPLE_DATABASE_URL` = Railway's `DATABASE_URL` **with the driver prefix swapped**:
     `postgresql://…` → `postgresql+psycopg://…`
   - `RIPPLE_REDIS_URL` = Railway's `REDIS_URL`
   - `RIPPLE_RERANKER_MODEL` — see "Shipping the fine-tuned reranker" below (or leave
     unset to serve without reranking; `?rerank=false` quality = bi-encoder).
4. Deploy. First boot: `/health` shows `"index": {"state": "empty"}`.
5. Index the demo repo (one-time): open a Railway shell on the service →
   `git clone --depth 1 https://github.com/pallets/flask /tmp/flask` → POST /index.
6. Sanity: `/health` (nodes > 0, cache available), `/` (demo UI), `/search?q=…`.

**Memory sizing:** embedder + reranker + FastAPI ≈ 1.5–2GB RSS. Pick a plan ≥ 2GB.

## Shipping the fine-tuned reranker

`models/reranker` is gitignored (91MB of weights). Two options:
- **Hugging Face Hub (recommended):** `huggingface-cli upload <you>/ripple-reranker
  models/reranker` (public), then set `RIPPLE_RERANKER_MODEL=<you>/ripple-reranker` —
  the container downloads it at first boot into `HF_HOME`.
- **Bake into the image:** remove `models` from `.dockerignore` and add
  `COPY models /app/models` to the Dockerfile; set the env to `/app/models/reranker`.

**Note on local container performance:** Docker Desktop on macOS runs a Linux VM —
inference and DB latency inside it are several times worse than both native macOS and
native Linux. Use it to verify *functionality*; trust the M7 numbers (native) and
re-benchmark on the cloud host for deployed performance.

## Cost & scale notes (measured, M7)

- The full rerank pipeline serves ~6 req/s per instance (cross-encoder-bound);
  `rerank=false` serves ~150 req/s and impact ~1,400 req/s. Fine for a demo.
- Redis cache absorbs repeat queries at ~5ms; the cache flushes on reindex.
- One instance is plenty; there is no multi-instance coordination for /index (last
  write wins) — documented limitation, acceptable for a single-operator demo.

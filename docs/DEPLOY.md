# Deploying Ripple — the free path (Hugging Face Spaces + Neon)

**Cost: $0.** Hugging Face Spaces (free Docker tier: 2 vCPU, 16GB RAM) hosts the app;
Neon (free tier) provides Postgres + pgvector; Redis is omitted (Ripple runs without it,
caching just turns off). The one tradeoff: both sleep when idle, so the first hit after a
quiet spell has a ~30–60s cold start while models load and, on the very first boot, the
demo repo indexes itself.

> Why not Railway/Render/Fly? An always-on ~2GB ML instance runs **~$20–30/mo** on those
> (idle containers still bill). HF Spaces' free tier has the RAM headroom and costs
> nothing. Trade a cold start for $0 — the right call for a portfolio link.

---

## Prerequisites (accounts — you create these; they're free, no card)

- A [Hugging Face](https://huggingface.co/join) account.
- A [Neon](https://neon.tech) account.

## Step 1 — Neon Postgres (~3 min)

1. New Project → note the connection string.
2. **Swap the driver prefix** for Ripple: `postgresql://…` → `postgresql+psycopg://…`
   (append `?sslmode=require` if not already present).
3. That's it — Ripple's `init_db()` runs `CREATE EXTENSION IF NOT EXISTS vector` and
   creates all tables on first connect. (0.5GB free storage; the Flask demo index is ~10MB.)

## Step 2 — (optional) publish the fine-tuned reranker (~2 min)

The reranker weights (`models/reranker`) are gitignored. To serve *with* reranking:

```bash
uv run huggingface-cli login
uv run huggingface-cli upload JamieXW/ripple-reranker models/reranker
```

Then set `RIPPLE_RERANKER_MODEL=JamieXW/ripple-reranker` in Step 3. Skip this to launch
faster; search still works (bi-encoder + graph expansion), just without the rerank stage.

## Step 3 — Hugging Face Space (~5 min)

1. **New Space** → SDK: **Docker** → blank template → 2 vCPU (free).
2. In the Space's **Settings → Variables and secrets**, add:
   - `RIPPLE_DATABASE_URL` = your Neon string from Step 1 (mark as **secret**)
   - `RIPPLE_AUTOINDEX_REPO` = `https://github.com/pallets/flask`
   - `RIPPLE_RERANKER_MODEL` = `JamieXW/ripple-reranker` (only if you did Step 2)
3. Push the Ripple repo to the Space, keeping the HF frontmatter as the Space's README:
   ```bash
   git clone https://huggingface.co/spaces/JamieXW/ripple hf-space && cd hf-space
   git remote add ripple https://github.com/JamieXW/Ripple-V1 && git fetch ripple
   git checkout ripple/main -- .            # bring in all the code
   cp deploy/huggingface/README.md README.md   # Space README with the frontmatter
   git add -A && git commit -m "deploy Ripple" && git push
   ```
4. HF builds the Dockerfile (~5–8 min first time). Watch the build log in the Space UI.

## Step 4 — first boot & verify

On first start the entrypoint sees an empty database, clones Flask, and indexes it
(~1–2 min) — you'll see `[entrypoint] first boot: indexing …` in the logs. Then:

- Space URL `/` → the demo UI (search + impact, live timings).
- `/health` → `"index": {"state": "ready", "nodes": 1661, …}`.
- `/search?q=load the session from a signed cookie` → cited results.

Subsequent boots find the index already in Neon and skip straight to serving.

## Notes

- **Cold start:** free Spaces sleep after 48h idle and wake on visit (~30–60s). The demo
  page shows a "waking up…" state and retries, so a first click doesn't look broken.
- **Local full-stack check** (no cloud): `docker compose --profile app up --build` runs
  the same image + Postgres + Redis; set `RIPPLE_AUTOINDEX_REPO` to watch it self-index.
  (Docker-on-Mac runs a Linux VM, so its latency is worse than native — trust M7's
  numbers for real performance.)
- **Redis (optional):** for cache hits on the Space, add a free [Upstash](https://upstash.com)
  Redis and set `RIPPLE_REDIS_URL`. Not required.

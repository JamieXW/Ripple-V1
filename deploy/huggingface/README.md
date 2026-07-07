---
title: Ripple
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: Hybrid graph + semantic code intelligence for Python
---

# Ripple (Hugging Face Space)

This is the deployment wrapper for [Ripple](https://github.com/JamieXW/Ripple-V1) — a
hybrid graph + semantic code-intelligence engine. The Space builds the repo's root
`Dockerfile` and serves the FastAPI app (demo UI at `/`, API at `/docs`).

**This file is the Space's README** — the YAML frontmatter above is what tells Hugging
Face to build the Docker image and route to port 8000. When you push the Ripple repo to
the Space, keep this frontmatter at the top of the Space's `README.md`.

## Required Space secrets/variables

| name | value |
|---|---|
| `RIPPLE_DATABASE_URL` | Neon connection string, driver-prefixed: `postgresql+psycopg://…` |
| `RIPPLE_AUTOINDEX_REPO` | `https://github.com/pallets/flask` (self-indexes on first boot) |
| `RIPPLE_RERANKER_MODEL` | your HF Hub reranker id, e.g. `JamieXW/ripple-reranker` (optional) |

Redis is intentionally omitted — Ripple runs without it (response caching just turns
off). See [docs/DEPLOY.md](https://github.com/JamieXW/Ripple-V1/blob/main/docs/DEPLOY.md)
for the full runbook.

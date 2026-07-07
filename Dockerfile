# Ripple API — production image (M8). Runs on Hugging Face Spaces (Docker) and locally.
# CPU-only torch on Linux (pinned via tool.uv.sources): serving inference runs CPU
# anyway (measured, M7), keeping the image at ~3.5GB instead of a measured 17.5GB with
# CUDA deps. Runs as a non-root user (HF Spaces requirement); `git` is present so the
# entrypoint can self-index a demo repo on first boot (see entrypoint.sh).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Non-root user with uid 1000 (HF Spaces runs containers as this user).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependency layer first (cached until pyproject/lock change).
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user src ./src
COPY --chown=user README.md LICENSE entrypoint.sh ./
RUN uv sync --frozen --no-dev

# Pre-bake the embedding model so first boot doesn't spend a minute downloading it.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=20 \
  CMD uv run --no-sync python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

ENTRYPOINT ["./entrypoint.sh"]

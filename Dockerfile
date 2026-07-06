# Ripple API — production image (M8).
# CPU-only torch on Linux (pinned via tool.uv.sources): serving inference runs CPU
# anyway (measured, M7), and it keeps the image at 3.5GB instead of a measured 17.5GB with CUDA deps.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1

# Dependency layer first (cached until pyproject/lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY README.md LICENSE ./
RUN uv sync --frozen --no-dev

# Pre-bake the embedding model so first boot doesn't spend a minute downloading.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=12 \
  CMD uv run --no-sync python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uv", "run", "--no-sync", "uvicorn", "ripple.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

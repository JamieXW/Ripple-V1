#!/usr/bin/env bash
# Ripple container entrypoint. Optionally self-indexes a demo repo on first boot,
# then serves the API. Set RIPPLE_AUTOINDEX_REPO to a git URL to enable auto-indexing
# (used by the Hugging Face Space so the demo provisions itself); leave it unset for
# local/compose use, where you index manually.
set -euo pipefail

if [ -n "${RIPPLE_AUTOINDEX_REPO:-}" ]; then
  # Only index if the database has no index yet (survives restarts / sleep-wake).
  needs_index=$(uv run --no-sync python - <<'PY' 2>/dev/null || echo yes
from ripple.db.repository import stored_model_name
from ripple.db.session import session_scope
try:
    with session_scope() as s:
        print("no" if stored_model_name(s) else "yes")
except Exception:
    print("yes")
PY
)
  if [ "$needs_index" = "yes" ]; then
    echo "[entrypoint] first boot: indexing $RIPPLE_AUTOINDEX_REPO …"
    rm -rf /tmp/autoindex
    git clone --depth 1 "$RIPPLE_AUTOINDEX_REPO" /tmp/autoindex
    uv run --no-sync ripple index /tmp/autoindex || echo "[entrypoint] indexing failed; serving empty"
  else
    echo "[entrypoint] index already present; skipping auto-index."
  fi
fi

exec uv run --no-sync uvicorn ripple.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"

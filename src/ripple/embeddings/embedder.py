"""Tier-0 embedding model wrapper (sentence-transformers).

The model is loaded lazily on first use so importing this module — and therefore
starting the CLI — stays cheap; ``torch`` is only imported when we actually embed.
Vectors are L2-normalized, so a dot product between them equals cosine similarity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ripple.config import settings


class Embedder:
    """Lazily-loaded sentence-transformers model that produces normalized vectors."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts (bulk indexing; GPU wins at this batch size)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=settings.bulk_embed_device or None,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query. Pinned to a serving device (CPU by default): at batch
        size 1, GPU launch overhead costs more than the compute saves — measured
        3ms cpu vs 6ms mps, and far worse under cross-thread serving contention."""
        vectors = self._load().encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=settings.query_device or None,
        )
        result: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32)[0]
        return result

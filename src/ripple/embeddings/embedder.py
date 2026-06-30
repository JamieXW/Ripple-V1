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
        """Embed a batch of texts into normalized row vectors."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query into one normalized vector."""
        vector: NDArray[np.float32] = self.encode([text])[0]
        return vector

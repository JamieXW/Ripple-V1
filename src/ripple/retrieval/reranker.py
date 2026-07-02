"""Cross-encoder reranking (M5b): the second stage of retrieve-then-rerank.

A cross-encoder reads (query, code) *together* in one input, so attention flows across
them — far more accurate than comparing two pre-computed embeddings, but too slow to run
over the whole corpus. So the bi-encoder retrieves a candidate set and this re-scores
only those. The model is loaded lazily (same rationale as ``Embedder``).
"""

from __future__ import annotations

from typing import Any, Protocol

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Anything that can score how well each text answers the query (higher = better)."""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """sentence-transformers ``CrossEncoder`` wrapper; accepts an HF name or a local path
    (e.g. our fine-tuned ``models/reranker``)."""

    def __init__(self, model_name_or_path: str = DEFAULT_RERANKER) -> None:
        self.model_name_or_path = model_name_or_path
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers.cross_encoder import CrossEncoder

            self._model = CrossEncoder(self.model_name_or_path)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict([(query, text) for text in texts])
        return [float(s) for s in scores]

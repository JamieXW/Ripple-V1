"""Training (M5b): fine-tune the cross-encoder reranker on mined pairs.
Train split only; the held-out test split grades the result."""

from ripple.training.train import build_training_examples, fine_tune_reranker

__all__ = ["build_training_examples", "fine_tune_reranker"]

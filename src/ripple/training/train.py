"""Fine-tune the cross-encoder reranker on mined pairs (M5b).

Builds examples strictly from the *train* split (the test split is the exam — never
touched here), then runs a hand-rolled PyTorch loop: tokenize (query, code) together,
BCE-with-logits on the relevance label, AdamW.

Why hand-rolled rather than sentence-transformers' ``fit()``: the trainer stack
(transformers Trainer + accelerate) auto-detects Apple's MPS backend and fought every
attempt to pin CPU — first an MPS autograd deadlock, then a model-on-CPU /
labels-on-MPS crash. An explicit loop makes device placement total and obvious, and at
this scale (~1k examples, 22M params) trains on CPU in minutes.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

from ripple.embeddings.vector_index import TextEncoder, VectorIndex
from ripple.mining.negatives import TrainingExample, mine_training_examples
from ripple.mining.pairs import mine_docstring_pairs, stripped_chunks
from ripple.parsing.models import ParsedModule
from ripple.retrieval.reranker import DEFAULT_RERANKER

logger = logging.getLogger(__name__)


def build_training_examples(
    modules: list[ParsedModule],
    repo_root: Path,
    encoder: TextEncoder,
    n_hard: int = 3,
    n_random: int = 1,
) -> list[TrainingExample]:
    """Mine (query, code, label) examples from the train split only."""
    pairs = [p for p in mine_docstring_pairs(modules, repo_root) if p.split == "train"]
    chunks = stripped_chunks(modules, repo_root)
    nodes = [node for node, _ in chunks]
    texts = [text for _, text in chunks]
    corpus = VectorIndex(nodes=nodes, matrix=encoder.encode(texts), model_name=encoder.model_name)
    query_matrix = encoder.encode([p.query for p in pairs])
    return mine_training_examples(
        pairs, corpus, texts, query_matrix, n_hard=n_hard, n_random=n_random
    )


def fine_tune_reranker(
    examples: list[TrainingExample],
    out_dir: Path,
    base_model: str = DEFAULT_RERANKER,
    epochs: int = 2,
    batch_size: int = 16,
    device: str = "cpu",
    learning_rate: float = 2e-5,
    seed: int = 13,
) -> Path:
    """Fine-tune ``base_model`` on ``examples`` and save the model to ``out_dir``."""
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    torch.manual_seed(seed)
    cross_encoder = CrossEncoder(base_model, num_labels=1, device=device)
    model = cross_encoder.model.to(device)
    tokenizer = cross_encoder.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    rng = random.Random(seed)
    order = list(range(len(examples)))
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = epochs * steps_per_epoch
    logger.info(
        "fine-tuning %s on %d examples (%d epochs, batch %d, %d steps, device %s)",
        base_model,
        len(examples),
        epochs,
        batch_size,
        total_steps,
        device,
    )

    model.train()
    step = 0
    for _epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[start : start + batch_size]]
            encoded = tokenizer(
                [e.query for e in batch],
                [e.code for e in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor([e.label for e in batch], device=device)
            logits = model(**encoded).logits.view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0 or step == total_steps:
                logger.info("step %d/%d loss %.4f", step, total_steps, loss.item())

    out_dir.mkdir(parents=True, exist_ok=True)
    cross_encoder.save(str(out_dir))
    return out_dir

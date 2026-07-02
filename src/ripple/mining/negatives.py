"""Training-example assembly: positives + mined hard negatives (M5b).

A cross-encoder learns fastest from *hard* negatives — candidates that look plausible
but are wrong. The best free source is the Tier-0 retriever itself: whatever it ranks
highly for a query that ISN'T the answer is, by construction, a convincing wrong answer.
We mix in a few random negatives so the model also sees easy contrast.

Only train-split pairs may enter here; the test split is the exam (see pairs.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ripple.embeddings.vector_index import VectorIndex
from ripple.mining.pairs import QueryCodePair


@dataclass(frozen=True)
class TrainingExample:
    """One supervised example for the cross-encoder: does ``code`` answer ``query``?"""

    query: str
    code: str
    label: float  # 1.0 relevant, 0.0 not


def mine_training_examples(
    train_pairs: list[QueryCodePair],
    corpus: VectorIndex,
    corpus_texts: list[str],
    query_matrix: NDArray[np.float32],
    n_hard: int = 3,
    n_random: int = 1,
    seed: int = 13,
) -> list[TrainingExample]:
    """Build labelled examples: each positive + retriever-mined hard negatives + random.

    ``corpus``/``corpus_texts`` must be parallel (same order); ``query_matrix`` holds the
    embedded queries of ``train_pairs`` row-for-row.
    """
    rng = random.Random(seed)
    text_by_qname = {node.qualified_name: corpus_texts[i] for i, node in enumerate(corpus.nodes)}
    all_qnames = [node.qualified_name for node in corpus.nodes]

    examples: list[TrainingExample] = []
    for i, pair in enumerate(train_pairs):
        examples.append(TrainingExample(query=pair.query, code=pair.code, label=1.0))

        # Hard negatives: the retriever's top-ranked wrong answers for this query.
        ranked = corpus.search(query_matrix[i], k=n_hard + 1)
        hard = [
            node.qualified_name for node, _ in ranked if node.qualified_name != pair.qualified_name
        ][:n_hard]
        for qname in hard:
            examples.append(TrainingExample(query=pair.query, code=text_by_qname[qname], label=0.0))

        # Random easy negatives for contrast.
        candidates = [q for q in all_qnames if q != pair.qualified_name and q not in hard]
        for qname in rng.sample(candidates, k=min(n_random, len(candidates))):
            examples.append(TrainingExample(query=pair.query, code=text_by_qname[qname], label=0.0))

    return examples

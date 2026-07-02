"""Grade semantic search against held-out docstring queries (M5a/M5b).

Each function's docstring is a free labelled query whose single correct answer is the
function itself. We build a corpus of ALL function/class chunks with docstrings stripped
(so a query can never match its own text — see mining/pairs.py on leakage), embed it,
run only the *test-split* queries, and score the ranked results.

With a ``reranker`` (M5b), each query retrieves ``retrieve_k`` candidates with the
bi-encoder and the cross-encoder reorders them before scoring — the same held-out
queries grade Tier-0 and the reranked pipeline, so the comparison is apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path
from statistics import fmean

from ripple.embeddings.vector_index import TextEncoder, VectorIndex
from ripple.eval.metrics import (
    SearchEvalReport,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ripple.mining.pairs import mine_docstring_pairs, stripped_chunks
from ripple.parsing.models import ParsedModule
from ripple.retrieval.reranker import Reranker


def build_eval_corpus(
    modules: list[ParsedModule], repo_root: Path, encoder: TextEncoder
) -> tuple[VectorIndex, list[str]]:
    """Embed every docstring-stripped chunk; returns the index and its parallel texts."""
    chunks = stripped_chunks(modules, repo_root)
    nodes = [node for node, _ in chunks]
    texts = [text for _, text in chunks]
    index = VectorIndex(nodes=nodes, matrix=encoder.encode(texts), model_name=encoder.model_name)
    return index, texts


def evaluate_search(
    modules: list[ParsedModule],
    repo_root: Path,
    encoder: TextEncoder,
    max_k: int = 10,
    reranker: Reranker | None = None,
    retrieve_k: int = 50,
) -> SearchEvalReport:
    """Run held-out docstring queries and aggregate ranking metrics (optionally reranked)."""
    pairs = mine_docstring_pairs(modules, repo_root)
    test_pairs = [p for p in pairs if p.split == "test"]
    n_train = sum(1 for p in pairs if p.split == "train")
    corpus, corpus_texts = build_eval_corpus(modules, repo_root, encoder)
    text_of = {node.qualified_name: corpus_texts[i] for i, node in enumerate(corpus.nodes)}

    if not test_pairs or corpus.matrix.shape[0] == 0:
        return SearchEvalReport(
            n_corpus=len(corpus.nodes),
            n_queries=0,
            n_train_pairs=n_train,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            ndcg_at_10=0.0,
        )

    query_matrix = encoder.encode([p.query for p in test_pairs])
    r1: list[float] = []
    r5: list[float] = []
    r10: list[float] = []
    rr: list[float] = []
    ndcg: list[float] = []
    for i, pair in enumerate(test_pairs):
        fetch_k = retrieve_k if reranker is not None else max_k
        candidates = [node.qualified_name for node, _ in corpus.search(query_matrix[i], k=fetch_k)]
        if reranker is not None:
            scores = reranker.score(pair.query, [text_of[q] for q in candidates])
            reordered = sorted(zip(candidates, scores, strict=False), key=lambda x: -x[1])
            candidates = [qname for qname, _ in reordered]
        ranked = candidates[:max_k]
        relevant = {pair.qualified_name}
        r1.append(recall_at_k(ranked, relevant, 1))
        r5.append(recall_at_k(ranked, relevant, 5))
        r10.append(recall_at_k(ranked, relevant, 10))
        rr.append(reciprocal_rank(ranked, relevant))
        ndcg.append(ndcg_at_k(ranked, relevant, 10))

    return SearchEvalReport(
        n_corpus=len(corpus.nodes),
        n_queries=len(test_pairs),
        n_train_pairs=n_train,
        recall_at_1=fmean(r1),
        recall_at_5=fmean(r5),
        recall_at_10=fmean(r10),
        mrr=fmean(rr),
        ndcg_at_10=fmean(ndcg),
    )

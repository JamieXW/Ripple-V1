"""Grade semantic search against held-out docstring queries (M5a).

Each function's docstring is a free labelled query whose single correct answer is the
function itself. We build a corpus of ALL function/class chunks with docstrings stripped
(so a query can never match its own text — see mining/pairs.py on leakage), embed it,
run only the *test-split* queries, and score the ranked results.

This produces the Tier-0 "before" number; M5b's reranker is measured against the exact
same held-out queries to show the lift.
"""

from __future__ import annotations

from pathlib import Path
from statistics import fmean

from ripple.embeddings.vector_index import TextEncoder, VectorIndex, iter_chunks
from ripple.eval.metrics import (
    SearchEvalReport,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ripple.mining.pairs import mine_docstring_pairs, strip_docstring
from ripple.parsing.models import ParsedModule


def build_eval_corpus(
    modules: list[ParsedModule], repo_root: Path, encoder: TextEncoder
) -> VectorIndex:
    """Embed every chunk with its docstring stripped (leakage-safe corpus)."""
    nodes = []
    texts = []
    for node, snippet in iter_chunks(modules, repo_root):
        if node.docstring is None:
            text: str | None = snippet
        else:
            text = strip_docstring(snippet)
        if text is None or not text.strip():
            continue  # docstring-only or unparseable — nothing retrievable
        nodes.append(node)
        texts.append(text)
    return VectorIndex(nodes=nodes, matrix=encoder.encode(texts), model_name=encoder.model_name)


def evaluate_search(
    modules: list[ParsedModule], repo_root: Path, encoder: TextEncoder, max_k: int = 10
) -> SearchEvalReport:
    """Run held-out docstring queries against the corpus and aggregate ranking metrics."""
    pairs = mine_docstring_pairs(modules, repo_root)
    test_pairs = [p for p in pairs if p.split == "test"]
    n_train = sum(1 for p in pairs if p.split == "train")
    corpus = build_eval_corpus(modules, repo_root, encoder)

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
        ranked = [node.qualified_name for node, _ in corpus.search(query_matrix[i], k=max_k)]
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

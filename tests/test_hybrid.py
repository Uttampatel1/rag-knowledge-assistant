"""Tests for BM25 lexical search, RRF fusion, and hybrid retrieval."""
from __future__ import annotations

from src.chunking import Chunk
from src.embeddings import HashingEmbedder
from src.hybrid import BM25, reciprocal_rank_fusion, tokenize
from src.vector_store import NumpyVectorStore


def _chunks(texts):
    return [Chunk(text=t, source="s.md", chunk_index=i) for i, t in enumerate(texts)]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Error XK-9 at Node_3!") == ["error", "xk", "9", "at", "node", "3"]


def test_bm25_ranks_document_with_query_term_first():
    corpus = [tokenize(t) for t in [
        "the cat sat on the mat",
        "distributed systems and consensus protocols",
        "raft is a consensus algorithm for replicated logs",
    ]]
    bm25 = BM25(corpus)
    ranked = bm25.rank("raft consensus", top_k=3)
    assert ranked[0] == 2  # the doc literally containing "raft"
    # a query term present in no document yields no positive-score hits
    assert bm25.rank("xylophone", top_k=3) == []


def test_rrf_rewards_items_ranked_high_in_multiple_lists():
    # id 2 is top of both lists -> must win; id 5 appears in only one list
    fused = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 5]], k=60)
    order = [idx for idx, _ in fused]
    assert order[0] == 2
    assert order.index(1) < order.index(5)  # 1 is in both lists, 5 in one


def test_keyword_search_finds_exact_token():
    emb = HashingEmbedder(dim=128)
    texts = ["quarterly revenue grew", "the XK9 thruster overheats", "customer onboarding guide"]
    store = NumpyVectorStore(dim=emb.dim)
    store.add(_chunks(texts), emb.embed(texts))
    hits = store.keyword_search("XK9 thruster", top_k=1)
    assert hits and hits[0].chunk.text == "the XK9 thruster overheats"


def test_hybrid_surfaces_lexical_match_and_runs():
    emb = HashingEmbedder(dim=128)
    texts = [
        "machine learning models for prediction",
        "return policy: refunds within 30 days",
        "error code E1042 means the disk is full",
    ]
    store = NumpyVectorStore(dim=emb.dim)
    store.add(_chunks(texts), emb.embed(texts))
    q = "E1042"
    results = store.search_hybrid(q, emb.embed_one(q), top_k=2, fetch_k=10)
    assert results, "hybrid search returned nothing"
    assert any("E1042" in r.chunk.text for r in results)
    # fused scores are positive RRF contributions
    assert all(r.score > 0 for r in results)


def test_bm25_index_rebuilds_after_add():
    emb = HashingEmbedder(dim=64)
    store = NumpyVectorStore(dim=emb.dim)
    store.add(_chunks(["alpha widget"]), emb.embed(["alpha widget"]))
    assert store.keyword_search("zeta", top_k=1) == []
    store.add(_chunks(["zeta gadget"]), emb.embed(["zeta gadget"]))
    hits = store.keyword_search("zeta", top_k=1)
    assert hits and hits[0].chunk.text == "zeta gadget"

from __future__ import annotations

import numpy as np

from src.chunking import Chunk
from src.vector_store import NumpyVectorStore


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _store():
    # A1 and A2 are near-duplicates; B is relevant but points elsewhere.
    vecs = np.vstack([
        _unit([0.95, 0.31, 0.0]),   # A1
        _unit([0.93, 0.36, 0.0]),   # A2 (almost identical to A1)
        _unit([0.80, 0.00, 0.60]),  # B  (different direction)
    ])
    chunks = [Chunk(text=t, source="s.md", chunk_index=i)
              for i, t in enumerate(["A1", "A2", "B"])]
    store = NumpyVectorStore(dim=3)
    store.add(chunks, vecs)
    return store


def test_plain_search_returns_near_duplicates():
    store = _store()
    q = _unit([1.0, 0.0, 0.0])
    texts = [r.chunk.text for r in store.search(q, top_k=2)]
    assert texts == ["A1", "A2"]   # the two near-duplicates crowd out B


def test_mmr_diversifies_away_from_duplicate():
    store = _store()
    q = _unit([1.0, 0.0, 0.0])
    texts = [r.chunk.text for r in store.search_mmr(q, top_k=2, lambda_mult=0.5)]
    assert texts[0] == "A1"        # most relevant still comes first
    assert "B" in texts            # but the redundant A2 is replaced by diverse B


def test_mmr_lambda_one_matches_pure_relevance():
    store = _store()
    q = _unit([1.0, 0.0, 0.0])
    plain = [r.chunk.text for r in store.search(q, top_k=3)]
    mmr = [r.chunk.text for r in store.search_mmr(q, top_k=3, lambda_mult=1.0)]
    assert mmr == plain


def test_mmr_empty_store_returns_empty():
    assert NumpyVectorStore(dim=3).search_mmr(np.zeros(3), top_k=2) == []

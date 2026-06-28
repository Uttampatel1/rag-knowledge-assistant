import numpy as np

from src.chunking import Chunk
from src.embeddings import HashingEmbedder
from src.vector_store import NumpyVectorStore


def _chunks(texts):
    return [Chunk(text=t, source="s.md", chunk_index=i) for i, t in enumerate(texts)]


def test_add_and_search_returns_most_similar():
    emb = HashingEmbedder(dim=128)
    texts = ["cats and dogs", "stock market trading", "machine learning models"]
    chunks = _chunks(texts)
    store = NumpyVectorStore(dim=emb.dim)
    store.add(chunks, emb.embed(texts))

    q = emb.embed_one("equities and trading strategies")
    results = store.search(q, top_k=1)
    assert results[0].chunk.text == "stock market trading"
    assert 0.0 <= results[0].score <= 1.0001


def test_dim_mismatch_raises():
    store = NumpyVectorStore(dim=10)
    bad = np.zeros((1, 5), dtype=np.float32)
    try:
        store.add(_chunks(["x"]), bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_and_load_roundtrip(tmp_path):
    emb = HashingEmbedder(dim=64)
    texts = ["alpha", "beta", "gamma"]
    store = NumpyVectorStore(dim=emb.dim)
    store.add(_chunks(texts), emb.embed(texts))
    store.save(str(tmp_path))

    assert NumpyVectorStore.exists(str(tmp_path))
    loaded = NumpyVectorStore.load(str(tmp_path))
    assert len(loaded) == 3
    assert loaded.sources() == ["s.md"]


def test_search_on_empty_store():
    store = NumpyVectorStore(dim=8)
    assert store.search(np.zeros(8, dtype=np.float32)) == []

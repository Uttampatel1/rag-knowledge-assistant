"""Pluggable text embedding backends.

Two backends are provided behind a common interface:

* :class:`SentenceTransformerEmbedder` — high-quality semantic embeddings
  (recommended for production). Requires ``sentence-transformers``.
* :class:`HashingEmbedder` — a deterministic, dependency-free fallback that
  hashes token n-grams into a fixed vector space. It needs no model download,
  so the whole system runs and tests pass offline.

``get_embedder()`` resolves the backend from settings, gracefully degrading to
the hashing embedder when ``sentence-transformers`` is unavailable.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from .config import Settings, get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    """Common interface for embedding backends."""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of L2-normalized vectors."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    @abstractmethod
    def name(self) -> str:
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder(Embedder):
    """Deterministic offline embedder using hashed word unigrams + bigrams."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                h = int.from_bytes(
                    hashlib.md5(feature.encode("utf-8")).digest()[:8], "little"
                )
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                vectors[row, idx] += sign
        return _l2_normalize(vectors)


class SentenceTransformerEmbedder(Embedder):
    """Wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._model_name = model_name

    @property
    def name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.astype(np.float32)


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Resolve the embedding backend from settings, degrading gracefully."""
    settings = settings or get_settings()
    backend = settings.embeddings_backend.lower()

    if backend == "hash":
        return HashingEmbedder()

    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception:  # noqa: BLE001 - any import/runtime failure -> fallback
            if backend == "sentence-transformers":
                raise
            return HashingEmbedder()

    raise ValueError(f"Unknown EMBEDDINGS_BACKEND: {settings.embeddings_backend}")

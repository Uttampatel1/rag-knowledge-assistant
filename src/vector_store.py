"""A small, dependency-free vector store with cosine similarity search.

The store keeps embeddings in a NumPy matrix and persists to a single ``.npz``
plus a JSON sidecar. The interface (``add`` / ``search`` / ``save`` / ``load``)
mirrors what you'd implement against Chroma or FAISS, so swapping in a managed
vector DB later is a localized change.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class NumpyVectorStore:
    """In-memory cosine-similarity store backed by a NumPy matrix."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != store dim {self.dim}"
            )
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if len(self._chunks) == 0:
            return []
        query = query_vector.astype(np.float32).reshape(-1)
        # Vectors are L2-normalized, so dot product == cosine similarity.
        scores = self._vectors @ query
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
        ]

    def search_mmr(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
    ) -> list[SearchResult]:
        """Maximal Marginal Relevance search — relevant *and* non-redundant.

        Plain top-k often returns near-duplicate chunks (the same fact phrased
        three ways), wasting the LLM's context window. MMR greedily picks the next
        chunk that maximises ``lambda * relevance - (1 - lambda) * redundancy``,
        where redundancy is the max similarity to anything already selected.
        ``lambda_mult=1`` reduces to pure relevance; ``0`` maximises diversity.
        """
        if len(self._chunks) == 0:
            return []
        query = query_vector.astype(np.float32).reshape(-1)
        scores = self._vectors @ query
        fetch_k = min(fetch_k, len(scores))
        top_k = min(top_k, fetch_k)
        candidates = list(np.argsort(-scores)[:fetch_k])

        selected: list[int] = []
        while candidates and len(selected) < top_k:
            if not selected:
                best = candidates.pop(0)
                selected.append(best)
                continue
            selected_mat = self._vectors[selected]
            best_idx, best_val = None, -np.inf
            for ci, cand in enumerate(candidates):
                redundancy = float(np.max(self._vectors[cand] @ selected_mat.T))
                mmr = lambda_mult * float(scores[cand]) - (1 - lambda_mult) * redundancy
                if mmr > best_val:
                    best_val, best_idx = mmr, ci
            selected.append(candidates.pop(best_idx))

        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in selected]

    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks})

    # --- persistence --------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(os.path.join(directory, "vectors.npz"), vectors=self._vectors)
        payload = [
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as fh:
            json.dump({"dim": self.dim, "chunks": payload}, fh, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "NumpyVectorStore":
        with open(os.path.join(directory, "chunks.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        store = cls(dim=data["dim"])
        store._vectors = np.load(os.path.join(directory, "vectors.npz"))["vectors"]
        store._chunks = [
            Chunk(
                text=c["text"],
                source=c["source"],
                chunk_index=c["chunk_index"],
                metadata=c.get("metadata", {}),
            )
            for c in data["chunks"]
        ]
        return store

    @staticmethod
    def exists(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, "chunks.json"))

"""Keyword (BM25) retrieval and Reciprocal Rank Fusion for hybrid search.

Dense vector search is great for *semantic* matches but can miss exact tokens —
acronyms, error codes, product SKUs, rare proper nouns — because those get washed
out in the embedding. Classic lexical BM25 nails them but ignores synonyms. Fusing
both with **Reciprocal Rank Fusion (RRF)** gives the best of each without tuning a
score-scale: RRF combines *ranks*, not raw scores, so the two retrievers don't need
to be calibrated against one another.

Pure-Python and dependency-free so it runs anywhere the rest of the project does.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over an in-memory token corpus."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in corpus]
        df: Counter = Counter()
        for d in corpus:
            for term in set(d):
                df[term] += 1
        # BM25 idf with the standard +0.5 smoothing (floored at a small positive).
        self.idf = {
            term: max(1e-6, math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        out = [0.0] * self.n
        for i in range(self.n):
            if not self.doc_len[i]:
                continue
            tf_i = self.tf[i]
            denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            s = 0.0
            for term in q_terms:
                f = tf_i.get(term, 0)
                if not f:
                    continue
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
            out[i] = s
        return out

    def rank(self, query: str, top_k: int) -> list[int]:
        """Indices of the top-k documents by BM25, dropping zero-score docs."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda t: -t[1])
        return [i for i, _ in scored[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several ranked id-lists into one, via RRF: score = sum 1/(k + rank).

    ``rankings`` is a list of ranked lists (best-first). Returns ``(id, score)``
    pairs sorted by fused score, descending.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda t: -t[1])

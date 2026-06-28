"""The end-to-end RAG pipeline: embed query -> retrieve -> generate -> cite.

This is the object the API and UI talk to. It owns the embedder, vector store,
and LLM provider, and returns answers together with the source passages used,
so every answer is auditable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .chunking import Chunk
from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .ingest import (
    build_store_from_chunks,
    chunks_from_text,
    ingest_directory,
)
from .llm_provider import LLMProvider, get_provider
from .vector_store import NumpyVectorStore


@dataclass
class Citation:
    marker: int
    source: str
    chunk_index: int
    score: float
    snippet: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "provider": self.provider,
            "citations": [c.__dict__ for c in self.citations],
        }


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        provider: LLMProvider | None = None,
        store: NumpyVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.provider = provider or get_provider(self.settings)
        self.store = store if store is not None else NumpyVectorStore(dim=self.embedder.dim)

    # --- index management ---------------------------------------------------
    def add_text(self, text: str, source: str) -> int:
        chunks = chunks_from_text(text, source=source, settings=self.settings)
        build_store_from_chunks(chunks, self.embedder, self.store)
        return len(chunks)

    def index_directory(self, directory: str) -> int:
        self.store = ingest_directory(directory, self.settings, self.embedder)
        return len(self.store)

    def save(self, directory: str | None = None) -> None:
        self.store.save(directory or self.settings.vector_store_dir)

    def load(self, directory: str | None = None) -> None:
        self.store = NumpyVectorStore.load(directory or self.settings.vector_store_dir)

    @property
    def num_chunks(self) -> int:
        return len(self.store)

    def sources(self) -> list[str]:
        return self.store.sources()

    # --- query --------------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None):
        top_k = top_k or self.settings.top_k
        query_vec = self.embedder.embed_one(question)
        return self.store.search(query_vec, top_k=top_k)

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        results = self.retrieve(question, top_k=top_k)
        contexts = [r.chunk.text for r in results]
        text = self.provider.generate(question, contexts)
        citations = [
            Citation(
                marker=i + 1,
                source=r.chunk.source,
                chunk_index=r.chunk.chunk_index,
                score=round(r.score, 4),
                snippet=_snippet(r.chunk),
            )
            for i, r in enumerate(results)
        ]
        return RAGAnswer(
            question=question,
            answer=text,
            citations=citations,
            provider=self.provider.name,
        )


def _snippet(chunk: Chunk, length: int = 220) -> str:
    text = chunk.text.strip().replace("\n", " ")
    return text if len(text) <= length else text[:length].rsplit(" ", 1)[0] + "…"


def load_or_build_default(settings: Settings | None = None) -> RAGPipeline:
    """Load a persisted index if present, otherwise ingest ``data/``."""
    settings = settings or get_settings()
    pipe = RAGPipeline(settings=settings)
    if NumpyVectorStore.exists(settings.vector_store_dir):
        pipe.load()
    elif os.path.isdir(settings.data_dir):
        pipe.index_directory(settings.data_dir)
    return pipe

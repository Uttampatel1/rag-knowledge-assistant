"""Document loading and index building.

Supports ``.pdf`` (via ``pypdf``), ``.txt`` and ``.md``. The public helpers
build a :class:`NumpyVectorStore` from raw text or from a directory of files.
"""
from __future__ import annotations

import os

from .chunking import Chunk, chunk_document
from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .vector_store import NumpyVectorStore

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def read_pdf(path: str) -> str:
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_text_file(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".txt", ".md"):
        return read_text_file(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunks_from_text(
    text: str,
    source: str,
    settings: Settings | None = None,
) -> list[Chunk]:
    settings = settings or get_settings()
    return chunk_document(
        text,
        source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def build_store_from_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    store: NumpyVectorStore | None = None,
) -> NumpyVectorStore:
    if store is None:
        store = NumpyVectorStore(dim=embedder.dim)
    if not chunks:
        return store
    vectors = embedder.embed([c.text for c in chunks])
    store.add(chunks, vectors)
    return store


def ingest_directory(
    directory: str,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> NumpyVectorStore:
    """Read every supported file in ``directory`` and build a vector store."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    store = NumpyVectorStore(dim=embedder.dim)

    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(entry)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = load_document(path)
        chunks = chunks_from_text(text, source=entry, settings=settings)
        build_store_from_chunks(chunks, embedder, store)

    return store

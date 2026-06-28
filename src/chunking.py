"""Text chunking utilities.

Documents are split into overlapping, sentence-aware chunks. Overlap preserves
context across boundaries so retrieval doesn't cut answers in half.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A retrievable unit of text plus provenance metadata."""

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}::chunk-{self.chunk_index}"


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines and trailing whitespace.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` chars.

    Splitting respects sentence boundaries where possible; a single sentence
    longer than ``chunk_size`` is hard-split as a last resort.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    # Clamp overlap to a sane fraction of chunk_size rather than failing hard,
    # so callers can shrink chunk_size without also tuning overlap.
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    text = _normalize(text)
    if not text:
        return []

    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            # Flush current, then hard-split the oversized sentence.
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), chunk_size - chunk_overlap):
                chunks.append(sentence[i : i + chunk_size].strip())
            continue

        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            chunks.append(current.strip())
            # Start the next chunk with a tail-overlap of the previous one.
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail} {sentence}".strip()

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def chunk_document(
    text: str,
    source: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Chunk a single document into :class:`Chunk` objects with provenance."""
    pieces = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    base_meta = metadata or {}
    return [
        Chunk(text=piece, source=source, chunk_index=i, metadata=dict(base_meta))
        for i, piece in enumerate(pieces)
    ]

"""Central configuration, loaded from environment / ``.env``.

Keeping configuration in one typed object makes the rest of the codebase free of
``os.getenv`` calls and easy to test (just construct a ``Settings`` instance).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Application settings resolved from the environment."""

    llm_provider: str = _get("LLM_PROVIDER", "mock")
    gemini_api_key: str = _get("GEMINI_API_KEY", "")
    gemini_model: str = _get("GEMINI_MODEL", "gemini-2.0-flash")

    embeddings_backend: str = _get("EMBEDDINGS_BACKEND", "auto")
    embedding_model: str = _get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_size: int = int(_get("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(_get("CHUNK_OVERLAP", "150"))
    top_k: int = int(_get("TOP_K", "4"))

    data_dir: str = _get("DATA_DIR", "data")
    vector_store_dir: str = _get("VECTOR_STORE_DIR", "data/vector_store")


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of :class:`Settings`."""
    return Settings()

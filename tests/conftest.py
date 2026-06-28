"""Shared test fixtures. Force offline backends so tests need no key/download."""
import os

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDINGS_BACKEND", "hash")

import pytest

from src.config import Settings
from src.rag_pipeline import RAGPipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="mock",
        embeddings_backend="hash",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
    )


@pytest.fixture
def pipeline(settings) -> RAGPipeline:
    pipe = RAGPipeline(settings=settings)
    pipe.add_text(
        "Northwind Analytics has three tiers: Starter, Growth, and Enterprise. "
        "Starter is free for up to 3 users. Growth costs 49 dollars per user per "
        "month. Enterprise adds SSO and audit logs.",
        source="pricing.md",
    )
    pipe.add_text(
        "Data is encrypted in transit using TLS and at rest using AES-256. "
        "Two-factor authentication is available on all tiers.",
        source="security.md",
    )
    return pipe

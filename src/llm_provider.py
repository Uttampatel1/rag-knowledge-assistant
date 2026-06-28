"""Pluggable LLM providers behind a single ``generate`` interface.

* :class:`MockProvider` — deterministic, offline, no API key. It produces a
  grounded answer by extracting the most relevant sentences from the retrieved
  context. Ideal for tests, demos, and CI.
* :class:`GeminiProvider` — Google Gemini. Requires ``GEMINI_API_KEY``.

Add a new provider by implementing :class:`LLMProvider` and registering it in
``get_provider``.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .config import Settings, get_settings

SYSTEM_PROMPT = (
    "You are a precise knowledge assistant. Answer the user's question using ONLY "
    "the provided context. Cite sources inline using [n] markers that refer to the "
    "numbered context passages. If the answer is not in the context, say you don't "
    "have enough information."
)


def build_prompt(question: str, contexts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context passages:\n{numbered}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with [n] citations):"
    )


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[str]) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MockProvider(LLMProvider):
    """Offline extractive provider — deterministic and key-free."""

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "I don't have enough information to answer that."

        q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored: list[tuple[float, int, str]] = []
        for idx, ctx in enumerate(contexts):
            for sentence in re.split(r"(?<=[.!?])\s+", ctx):
                terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                if not terms:
                    continue
                overlap = len(q_terms & terms) / (len(q_terms) + 1e-9)
                if overlap > 0:
                    scored.append((overlap, idx, sentence.strip()))

        scored.sort(key=lambda x: -x[0])
        top = scored[:3]
        if not top:
            # Fall back to the first passage so the answer is always grounded.
            return f"{contexts[0].strip()[:300]} [1]"

        seen: set[str] = set()
        parts: list[str] = []
        for _, idx, sentence in top:
            if sentence in seen:
                continue
            seen.add(sentence)
            parts.append(f"{sentence} [{idx + 1}]")
        return " ".join(parts)


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    @property
    def name(self) -> str:
        return self._model_name

    def generate(self, question: str, contexts: list[str]) -> str:
        prompt = build_prompt(question, contexts)
        response = self._model.generate_content(prompt)
        return (response.text or "").strip()


def get_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockProvider()
    if provider == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")

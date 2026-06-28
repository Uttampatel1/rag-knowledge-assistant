from src.llm_provider import MockProvider, build_prompt


def test_pipeline_answers_with_citations(pipeline):
    result = pipeline.answer("How much does the Growth tier cost?")
    assert result.citations, "expected at least one citation"
    # Retrieval should surface the pricing document.
    assert any(c.source == "pricing.md" for c in result.citations)
    assert "49" in result.answer or "Growth" in result.answer


def test_pipeline_retrieval_ranks_relevant_source(pipeline):
    results = pipeline.retrieve("encryption and two-factor authentication", top_k=1)
    assert results[0].chunk.source == "security.md"


def test_mock_provider_is_grounded_and_deterministic():
    provider = MockProvider()
    contexts = ["The sky is blue.", "Water boils at 100 degrees Celsius."]
    a1 = provider.generate("What color is the sky?", contexts)
    a2 = provider.generate("What color is the sky?", contexts)
    assert a1 == a2
    assert "sky" in a1.lower()
    assert "[1]" in a1 or "[2]" in a1


def test_mock_provider_handles_no_context():
    assert "enough information" in MockProvider().generate("anything", []).lower()


def test_build_prompt_numbers_contexts():
    prompt = build_prompt("Q?", ["first", "second"])
    assert "[1] first" in prompt
    assert "[2] second" in prompt


def test_answer_to_dict_shape(pipeline):
    d = pipeline.answer("What is the free tier?").to_dict()
    assert {"question", "answer", "provider", "citations"} <= d.keys()
    assert isinstance(d["citations"], list)

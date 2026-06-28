from src.chunking import chunk_document, split_text


def test_split_text_respects_chunk_size():
    text = " ".join(f"Sentence number {i} here." for i in range(200))
    chunks = split_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 + 40 for c in chunks)


def test_split_text_overlap_preserves_context():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    chunks = split_text(text, chunk_size=40, chunk_overlap=15)
    assert len(chunks) >= 2


def test_split_text_handles_oversized_sentence():
    text = "x" * 1000
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_document_sets_provenance():
    chunks = chunk_document("One. Two. Three.", "doc.md", chunk_size=100)
    assert chunks[0].source == "doc.md"
    assert chunks[0].chunk_index == 0
    assert chunks[0].id == "doc.md::chunk-0"


def test_empty_text_returns_no_chunks():
    assert split_text("   ") == []

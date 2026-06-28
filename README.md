# 📚 RAG Knowledge Assistant — Document Q&A with Cited Sources

A production-style **Retrieval-Augmented Generation** service that ingests your PDFs and text, then answers natural-language questions **grounded in those documents, with inline source citations**. Ships as a FastAPI API *and* a Streamlit app.

> Built for teams that want "ChatGPT over our own documents" without hallucinations — every answer points back to the passages it came from.

---

## The problem it solves

Generic LLMs don't know your internal handbooks, contracts, product docs, or policies — and when asked, they confidently make things up. This service grounds an LLM in **your** corpus: it retrieves the most relevant passages and forces the model to answer from them, citing each source. That turns an unverifiable chatbot into an **auditable** knowledge tool.

## Key capabilities (what it proves)

- **End-to-end RAG pipeline:** ingest → chunk → embed → vector search → grounded generation → citations.
- **Diversity-aware retrieval:** optional **Maximal Marginal Relevance (MMR)** re-ranking (`USE_MMR=true`) trades a little relevance for less-redundant context, so the LLM doesn't get the same fact three times.
- **Pluggable LLM provider:** Google **Gemini** in production; a deterministic **offline mock** provider for tests/CI/demos (no API key needed).
- **Pluggable embeddings:** `sentence-transformers` for semantic quality, with a dependency-free **hashing fallback** so the system runs fully offline.
- **Auditable answers:** each response returns the exact passages used, with similarity scores.
- **Two interfaces:** documented REST API (FastAPI/OpenAPI) and a Streamlit UI.
- **Production-ready:** structured logging (`LOG_LEVEL`), `Dockerfile` + `docker-compose.yml`, and GitHub Actions CI.
- **Tested:** 22 `pytest` tests covering chunking, vector search (plain + MMR), persistence, and the pipeline.

## Demo

**Sample run** (offline mock provider, against the bundled synthetic knowledge base):

```text
Q: How long are raw events retained?
A: By default, raw events are retained for 13 months. [1]
   Aggregated metrics are retained indefinitely unless deleted. [1]
Sources: [1] security_and_data.md  [2] getting_started.md  ...
```

**API example:**

```bash
# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What payment methods are accepted?", "top_k": 4}'
```

```json
{
  "question": "What payment methods are accepted?",
  "answer": "We accept all major credit cards. Enterprise customers can pay by invoice... [1]",
  "provider": "mock",
  "citations": [
    {"marker": 1, "source": "billing_faq.md", "chunk_index": 0, "score": 0.71, "snippet": "..."}
  ]
}
```

```bash
# Upload a document
curl -X POST http://localhost:8000/ingest/file -F "file=@handbook.pdf"
```

> **Note on offline mode:** with the hashing embedder + mock provider (the zero-setup defaults), retrieval is keyword-based and answers are extractive — perfect for demos and CI. For semantic retrieval and fluent answers, set `EMBEDDINGS_BACKEND=sentence-transformers` and `LLM_PROVIDER=gemini`.

## How it works

```
                 ┌─────────────┐
   PDF/TXT/MD ─► │   Ingest    │ ─► chunk (overlapping, sentence-aware)
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐     embeddings (sentence-transformers | hashing)
                 │  Embed +    │ ─────────────────────────────────────────────┐
                 │  Index      │                                               │
                 └─────────────┘                                               ▼
                                                                      ┌─────────────────┐
   user question ─► embed query ─► cosine search (top-k) ──────────►  │  Vector Store   │
                                          │                           └─────────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │  LLM (Gemini | mock)    │ ─► grounded answer + [n] citations
                              │  "answer ONLY from      │
                              │   context, cite [n]"    │
                              └────────────────────────┘
```

The `VectorStore` interface (`add` / `search` / `save` / `load`) mirrors what you'd build against Chroma or FAISS, so swapping the default NumPy store for a managed vector DB is a localized change.

## Tech stack

- **API:** FastAPI, Uvicorn, Pydantic
- **Retrieval:** NumPy cosine-similarity vector store (Chroma/FAISS-compatible interface)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`) with hashing fallback
- **LLM:** Google Gemini (`google-generativeai`), pluggable via an `LLMProvider` interface
- **UI:** Streamlit
- **Docs:** pypdf for PDF extraction
- **Tests:** pytest

## Setup & run

```bash
cd 01-rag-knowledge-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # defaults work offline, no key needed

# 1) Generate the sample knowledge base
python -m src.generate_data

# 2a) Run the API
uvicorn api:app --reload
#    -> open http://localhost:8000/docs

# 2b) ...or run the Streamlit app
streamlit run app.py

# Run tests (offline; no key/model download)
pytest -q
```

**Enable production mode** (semantic retrieval + Gemini) in `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
EMBEDDINGS_BACKEND=sentence-transformers
```

## Project structure

```
01-rag-knowledge-assistant/
├── api.py                  # FastAPI service (ingest, query, reindex)
├── app.py                  # Streamlit UI (standalone, calls pipeline directly)
├── src/
│   ├── config.py           # typed settings from .env
│   ├── chunking.py         # overlapping, sentence-aware text splitter
│   ├── embeddings.py       # sentence-transformers + hashing fallback
│   ├── vector_store.py     # NumPy cosine store (Chroma/FAISS-style) + MMR search
│   ├── ingest.py           # PDF/TXT/MD loading + index building
│   ├── llm_provider.py     # pluggable Gemini / mock providers
│   ├── logging_utils.py    # structured logging + timing
│   ├── rag_pipeline.py     # retrieve (plain/MMR) → generate → cite orchestration
│   └── generate_data.py    # synthetic knowledge base generator
├── tests/                  # 22 pytest tests (offline)
├── Dockerfile              # containerised FastAPI service
├── docker-compose.yml
├── .github/workflows/ci.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

## Possible extensions

- Swap the NumPy store for **Chroma/FAISS/pgvector** (interface is already isolated).
- **Hybrid retrieval** (BM25 + dense) and a cross-encoder **re-ranker** for higher precision.
- **Streaming** token responses and conversational memory for multi-turn Q&A.
- **Evaluation harness** (faithfulness / answer-relevancy) to track quality across changes.
- Per-document **access control** and multi-tenant workspaces.

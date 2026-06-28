"""FastAPI service for the RAG Knowledge Assistant.

Endpoints
---------
GET  /health                 -> liveness + index stats
POST /ingest/text            -> add a raw text document to the index
POST /ingest/file            -> upload a .pdf/.txt/.md file
POST /query                  -> ask a question, get an answer + citations
POST /reindex                -> rebuild the index from the data directory

Run:  uvicorn api:app --reload
"""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import get_settings
from src.ingest import load_document
from src.rag_pipeline import RAGPipeline, load_or_build_default

settings = get_settings()
app = FastAPI(
    title="RAG Knowledge Assistant",
    version="1.0.0",
    description="Document Q&A with cited sources. LLM provider is pluggable.",
)

pipeline: RAGPipeline = load_or_build_default(settings)


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field("inline.txt", description="Logical name for citations")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": pipeline.provider.name,
        "embedder": pipeline.embedder.name,
        "num_chunks": pipeline.num_chunks,
        "sources": pipeline.sources(),
    }


@app.post("/ingest/text")
def ingest_text(req: TextIngestRequest) -> dict:
    added = pipeline.add_text(req.text, source=req.source)
    return {"added_chunks": added, "num_chunks": pipeline.num_chunks}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {ext}")
    suffix = ext or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = load_document(tmp_path)
    finally:
        os.unlink(tmp_path)
    added = pipeline.add_text(text, source=file.filename or f"upload{suffix}")
    return {"filename": file.filename, "added_chunks": added, "num_chunks": pipeline.num_chunks}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if pipeline.num_chunks == 0:
        raise HTTPException(409, "Index is empty. Ingest documents first.")
    return pipeline.answer(req.question, top_k=req.top_k).to_dict()


@app.post("/reindex")
def reindex() -> dict:
    if not os.path.isdir(settings.data_dir):
        raise HTTPException(404, f"Data dir not found: {settings.data_dir}")
    count = pipeline.index_directory(settings.data_dir)
    return {"num_chunks": count, "sources": pipeline.sources()}

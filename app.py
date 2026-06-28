"""Streamlit front end for the RAG Knowledge Assistant.

Talks to the pipeline directly (no running API required) so it works as a
standalone demo. Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import get_settings
from src.generate_data import write_sample_docs
from src.rag_pipeline import RAGPipeline, load_or_build_default

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return load_or_build_default(get_settings())


pipe = get_pipeline()

st.title("📚 RAG Knowledge Assistant")
st.caption(
    f"Provider: **{pipe.provider.name}** · Embedder: **{pipe.embedder.name}** · "
    f"Indexed chunks: **{pipe.num_chunks}**"
)

with st.sidebar:
    st.header("Knowledge base")
    if st.button("Load sample documents"):
        write_sample_docs(get_settings().data_dir)
        pipe.index_directory(get_settings().data_dir)
        st.success(f"Indexed {pipe.num_chunks} chunks.")
    st.write("**Sources:**")
    for s in pipe.sources() or ["(none yet)"]:
        st.write(f"- {s}")

    st.divider()
    uploaded = st.file_uploader("Add a document", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("Ingest uploaded file"):
        import tempfile, os
        from src.ingest import load_document

        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        text = load_document(tmp_path)
        os.unlink(tmp_path)
        added = pipe.add_text(text, source=uploaded.name)
        st.success(f"Added {added} chunks from {uploaded.name}.")

question = st.text_input("Ask a question about your documents:")
top_k = st.slider("Passages to retrieve (top-k)", 1, 10, get_settings().top_k)

if question:
    if pipe.num_chunks == 0:
        st.warning("Index is empty. Load sample documents or upload a file first.")
    else:
        result = pipe.answer(question, top_k=top_k)
        st.subheader("Answer")
        st.write(result.answer)
        st.subheader("Sources")
        for c in result.citations:
            with st.expander(f"[{c.marker}] {c.source} (score {c.score})"):
                st.write(c.snippet)

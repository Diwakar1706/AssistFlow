"""
MAIN APP (browser UI) — Upload a PDF, ask questions in normal human language.

Run:
  streamlit run app_streamlit.py
Then in the browser: upload your PDF (even 100+ pages) -> wait for indexing -> ask questions.
Needs OPENAI_API_KEY in a .env file next to this script (copy .env.example to .env).
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import PdfChat, get_vector_store  # noqa: E402

st.set_page_config(page_title="PDF Q&A — RAG for Beginners", page_icon="📄")
st.title("📄 Ask questions about your PDF")
st.caption("Upload any PDF — a 100+ page policy, report or book — and ask in plain English. Answers come only from your PDF, with page numbers.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Copy .env.example to .env, put your OpenAI key in it, and restart.")
    st.stop()

uploaded = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded is not None:
    if st.session_state.get("file_name") != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        with st.spinner(f"Indexing {uploaded.name} — a large PDF can take a minute the first time..."):
            store, n_pages, n_chunks, reused = get_vector_store(tmp_path)
        st.session_state.update(file_name=uploaded.name, chat=PdfChat(store), msgs=[])
        st.success(f"Ready — {('reused index, ' if reused else '')}{n_chunks} chunks indexed" + (f" from {n_pages} pages." if n_pages else "."))

if "chat" in st.session_state:
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    q = st.chat_input("Ask a question about this PDF...")
    if q:
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF..."):
                answer, pages, _ = st.session_state.chat.ask(q)
            st.write(answer)
            st.caption(f"Based on page(s): {pages}")
        st.session_state.msgs.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF above to start.")

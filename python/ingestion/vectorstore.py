"""
Vector store management — embed document chunks with OpenAI and persist
them in per-document ChromaDB stores on disk (one store per document).
"""
import hashlib
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import DB_ROOT, EMBED_MODEL
from ingestion.chunking import load_and_split_pdf


def pdf_id(pdf_path: str) -> str:
    """Stable, filesystem-safe id for a document file (name + first-MB hash).
    Each document gets its own ChromaDB folder under db/<pdf_id>/."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read(1024 * 1024))
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40]
    return f"{stem}_{h.hexdigest()[:8]}"


def get_vector_store(pdf_path: str, force_rebuild: bool = False, extra_chunks=None):
    """Create (or reuse) the ChromaDB store for this document.

    Returns (vectorstore, num_pages, num_chunks, reused).

    extra_chunks — optional additional Document objects (e.g. GPT-4o image
    descriptions) that are embedded together with the text chunks.  Passing
    them forces a rebuild so the extras are definitely included.
    """
    persist_dir = os.path.join(DB_ROOT, pdf_id(pdf_path))
    embeddings  = OpenAIEmbeddings(model=EMBED_MODEL)

    if extra_chunks:
        force_rebuild = True

    # Reuse existing store if it has chunks and no rebuild is requested
    if os.path.exists(persist_dir) and not force_rebuild:
        store = Chroma(persist_directory=persist_dir,
                       embedding_function=embeddings,
                       collection_metadata={"hnsw:space": "cosine"})
        count = store._collection.count()
        if count > 0:
            return store, None, count, True

    # Build a fresh store
    pages, chunks = load_and_split_pdf(pdf_path)
    if extra_chunks:
        chunks = chunks + list(extra_chunks)

    store = None
    batch = 100   # embed in batches so large PDFs don't hit API limits in one call
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        if store is None:
            store = Chroma.from_documents(
                documents=part,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"})
        else:
            store.add_documents(part)
        print(f"  embedded chunks {min(i + batch, len(chunks))}/{len(chunks)}")

    return store, len(pages), len(chunks), False

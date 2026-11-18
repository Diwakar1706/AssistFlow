"""
Hybrid search — combine vector similarity (ChromaDB) with BM25 keyword
search using LangChain's EnsembleRetriever (70 / 30 weighting).
"""
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_community.retrievers import EnsembleRetriever   # type: ignore

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.retrievers import BM25Retriever   # type: ignore


def build_hybrid_retriever(vectorstore, docs, candidate_k: int):
    """Build an EnsembleRetriever (vector + BM25) over the document chunks.
    Returns the retriever, or None if BM25 setup fails (pure-vector fallback
    kicks in automatically in retrieve_candidates)."""
    if not docs:
        return None
    try:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = candidate_k
        vector = vectorstore.as_retriever(search_kwargs={"k": candidate_k})
        return EnsembleRetriever(retrievers=[vector, bm25], weights=[0.7, 0.3])
    except Exception:
        return None


def retrieve_candidates(hybrid_retriever, vectorstore, question: str, candidate_k: int):
    """Retrieve top candidate chunks.
    Uses hybrid search if available, falls back to vector-only.
    Returns (documents, mode_string)."""
    if hybrid_retriever is not None:
        try:
            docs = hybrid_retriever.invoke(question)
            return docs, "hybrid (vector + BM25)"
        except Exception:
            pass
    retriever = vectorstore.as_retriever(search_kwargs={"k": candidate_k})
    return retriever.invoke(question), "vector only"

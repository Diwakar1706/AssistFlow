"""
Conversational RAG engine — orchestrates follow-up question condensation,
hybrid retrieval, reranking, and answer generation (with SSE streaming).
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import CHAT_MODEL
from retrieval.search import build_hybrid_retriever, retrieve_candidates
from retrieval.reranker import rerank


class PdfChat:
    """Ask plain-English questions about one indexed document.
    Keeps a short rolling chat history (last 3 turns).

    Pipeline per question
    ─────────────────────
    1. _condense_question  — rewrite follow-ups into standalone queries
    2. retrieve_candidates — hybrid vector + BM25 search  → top 12 chunks
    3. rerank              — Cohere or LLM listwise rerank → top 6 chunks
    4. ask / ask_stream    — GPT answers ONLY from those 6 chunks
    """

    def __init__(self, vectorstore, docs=None,
                 model: str = CHAT_MODEL, k: int = 6, candidate_k: int = 12):
        self.db           = vectorstore
        self.llm          = ChatOpenAI(model=model, temperature=0)
        self.k            = k
        self.candidate_k  = candidate_k
        self.history      = []
        self.last_mode    = "vector"
        self.last_query_used = None
        self._hybrid      = build_hybrid_retriever(vectorstore, docs, candidate_k)

    # ── Step 0 — follow-up rewriting ──────────────────────────────────
    def _condense_question(self, question: str) -> str:
        """Rewrite a follow-up into a standalone question using history.
        'and what about its cost?' → 'What is the premium for the family floater plan?'
        Returns the question unchanged if no history or already standalone."""
        if not self.history:
            return question
        try:
            convo = "\n".join(
                ("User: " if isinstance(m, HumanMessage) else "Assistant: ")
                + (m.content if isinstance(m.content, str) else str(m.content))
                for m in self.history[-6:]
            )
            prompt = (
                f"Conversation so far:\n{convo}\n\nFollow-up question: {question}\n\n"
                "Rewrite the follow-up as a complete standalone question containing every "
                "detail needed to search a document for its answer (names, plan, topic from "
                "the conversation). If it is already standalone, return it unchanged. "
                "Return ONLY the question — no quotes, no explanation."
            )
            res  = self.llm.invoke([HumanMessage(content=prompt)])
            text = res.content if isinstance(res.content, str) else str(res.content)
            text = text.strip().strip('"').strip()
            return text if text else question
        except Exception:
            return question

    # ── Steps 1 + 2 — retrieve and rerank ─────────────────────────────
    def retrieve(self, question: str) -> list:
        """Run hybrid search then reranking; sets self.last_mode."""
        candidates, mode   = retrieve_candidates(
            self._hybrid, self.db, question, self.candidate_k)
        reranked, suffix   = rerank(question, candidates, self.k, self.llm)
        self.last_mode     = mode + suffix
        return reranked

    # ── Prompt building ────────────────────────────────────────────────
    def _answer_messages(self, question: str, docs: list) -> list:
        context = "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
        )
        prompt = (
            f"Answer this question using ONLY the document excerpts below.\n"
            f"Write in simple, clear human language. Cite the page number(s) you used, "
            f"like (Page 12).\n"
            f"If the answer is not in these excerpts, say exactly: "
            f'"I don\'t have enough information to answer that question based on the provided documents."\n\n'
            f"Question: {question}\n\nDocument excerpts:\n{context}\n"
        )
        return (
            [SystemMessage(content="You are a helpful assistant that answers questions "
                                   "strictly from the provided PDF document.")]
            + self.history
            + [HumanMessage(content=prompt)]
        )

    def _remember(self, question: str, answer: str):
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        self.history = self.history[-6:]   # keep last 3 turns only

    # ── Step 3a — full answer ──────────────────────────────────────────
    def ask(self, question: str):
        """Full pipeline — returns (answer, pages_used, docs)."""
        standalone = self._condense_question(question)
        self.last_query_used = standalone
        docs   = self.retrieve(standalone)
        result = self.llm.invoke(self._answer_messages(question, docs))
        answer = result.content if isinstance(result.content, str) else str(result.content)
        self._remember(question, answer)
        pages  = sorted({d.metadata.get("page", "?") for d in docs}, key=lambda x: str(x))
        return answer, pages, docs

    # ── Step 3b — streaming answer (SSE) ──────────────────────────────
    def ask_stream(self, question: str):
        """Same pipeline as ask(), but yields SSE-ready events:
            {type: 'context', docs, pages, mode, searched_as}
            {type: 'token',   text}   ← many of these
            {type: 'done',    answer}
        """
        standalone = self._condense_question(question)
        self.last_query_used = standalone
        docs  = self.retrieve(standalone)
        pages = sorted({d.metadata.get("page", "?") for d in docs}, key=lambda x: str(x))
        yield {
            "type":        "context",
            "docs":        docs,
            "pages":       pages,
            "mode":        self.last_mode,
            "searched_as": standalone if standalone != question else None,
        }
        parts = []
        for chunk in self.llm.stream(self._answer_messages(question, docs)):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                parts.append(text)
                yield {"type": "token", "text": text}
        answer = "".join(parts)
        self._remember(question, answer)
        yield {"type": "done", "answer": answer}

"""
Reranking — re-order retrieved candidates by relevance.
Tries Cohere Rerank first (if COHERE_API_KEY is set), then falls back
to an LLM listwise rerank using the same OpenAI key already in use.
"""
import json
import os
import re

from langchain_core.messages import HumanMessage


def rerank(question: str, candidates: list, k: int, llm):
    """Rerank candidate chunks and return the top-k.
    Returns (reranked_docs, mode_suffix_string)."""
    if len(candidates) <= k:
        return candidates, ""

    # ── 1. Cohere Rerank (optional — only if COHERE_API_KEY is set) ───
    if os.getenv("COHERE_API_KEY"):
        try:
            from langchain_cohere import CohereRerank
            reranker = CohereRerank(model="rerank-english-v3.0", top_n=k)
            out = reranker.compress_documents(candidates, question)
            if out:
                return out, " + Cohere rerank"
        except Exception:
            pass

    # ── 2. LLM listwise rerank (no extra key needed) ──────────────────
    try:
        listing = "\n\n".join(
            f"[{i}] (Page {d.metadata.get('page', '?')}) {d.page_content[:250]}"
            for i, d in enumerate(candidates)
        )
        prompt = (
            f"Question: {question}\n\nPassages:\n{listing}\n\n"
            f"Return ONLY a JSON list of the passage numbers of the {k} most relevant "
            "passages, most relevant first, e.g. [3, 0, 5]. No other text."
        )
        res  = llm.invoke([HumanMessage(content=prompt)])
        text = res.content if isinstance(res.content, str) else str(res.content)
        m    = re.search(r"\[[0-9,\s]+\]", text)
        order = json.loads(m.group(0)) if m else []
        picked, seen = [], set()
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                picked.append(candidates[idx])
                seen.add(idx)
            if len(picked) >= k:
                break
        if picked:
            return picked, " + LLM rerank"
    except Exception:
        pass

    return candidates[:k], ""

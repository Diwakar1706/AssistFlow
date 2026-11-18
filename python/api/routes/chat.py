"""
Chat routes — synchronous chat and SSE streaming chat endpoints.
"""
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.messages import AIMessage as _AI, HumanMessage as _HU

from api.helpers import sources_for
from api.models import ConvChatIn
from api.state import _conv_chats, _chats
from chat.engine import PdfChat
from ingestion.vectorstore import get_vector_store
from storage.conversations import conv_summary, list_convs, load_conv, save_conv
from storage.documents import load_doc

router = APIRouter()


def _get_conv_chat(conv: dict) -> PdfChat:
    """Return (or create) a PdfChat for a conversation, with its history
    seeded from the last 3 saved turns so follow-ups still work after restart."""
    if conv["id"] in _conv_chats:
        return _conv_chats[conv["id"]]

    doc   = load_doc(conv["document_id"])
    store, _, _, _ = get_vector_store(doc["pdf_path"])
    chunks_as_docs = [
        Document(page_content=c["content"],
                 metadata={"page": c["page"],
                            "source_file": doc["name"],
                            "chunk_id":    c["id"]})
        for c in doc.get("chunks", [])
    ]
    chat_obj = PdfChat(store, docs=chunks_as_docs)
    history  = []
    for m in conv.get("messages", [])[-6:]:
        history.append(_HU(content=m["content"]) if m["role"] == "user"
                       else _AI(content=m["content"]))
    chat_obj.history = history
    _conv_chats[conv["id"]] = chat_obj
    return chat_obj


# ── Synchronous chat ───────────────────────────────────────────────────

@router.post("/conversations/{conv_id}/chat")
def chat_in_conversation(conv_id: str, body: ConvChatIn):
    conv = load_conv(conv_id)
    doc  = load_doc(conv["document_id"])
    if doc["steps"].get("vectorization") != "done":
        raise HTTPException(400, doc.get("vector_note") or "This document is not vectorized yet.")

    chat_obj = _get_conv_chat(conv)
    answer, pages, used_docs = chat_obj.ask(body.question)
    sources  = sources_for(doc, used_docs)
    searched = (chat_obj.last_query_used
                if chat_obj.last_query_used and chat_obj.last_query_used != body.question
                else None)
    now = int(time.time())
    conv["messages"].append({"role": "user",      "content": body.question, "ts": now})
    conv["messages"].append({"role": "assistant",  "content": answer,
                              "pages": pages, "retrieval": chat_obj.last_mode,
                              "sources": sources, "searched_as": searched, "ts": now})
    if conv["title"].startswith("Chat ") and len(conv["messages"]) == 2:
        conv["title"] = body.question[:42]
    conv["updated_at"] = now
    save_conv(conv)
    return {"answer": answer, "pages": pages, "retrieval": chat_obj.last_mode,
            "sources": sources, "searched_as": searched,
            "conversation": conv_summary(conv)}


# ── SSE streaming chat ─────────────────────────────────────────────────

@router.post("/conversations/{conv_id}/chat/stream")
def chat_stream(conv_id: str, body: ConvChatIn):
    """Streaming version — Server-Sent Events.
    Events: meta → token (many) → done | error"""
    conv = load_conv(conv_id)
    doc  = load_doc(conv["document_id"])
    if doc["steps"].get("vectorization") != "done":
        raise HTTPException(400, doc.get("vector_note") or "This document is not vectorized yet.")

    chat_obj = _get_conv_chat(conv)

    def event_stream():
        sources, searched, mode, pages = [], None, "", []
        try:
            for ev in chat_obj.ask_stream(body.question):
                if ev["type"] == "context":
                    pages, mode, searched = ev["pages"], ev["mode"], ev["searched_as"]
                    sources = sources_for(doc, ev["docs"])
                    yield "data: " + json.dumps({
                        "type": "meta", "pages": pages, "retrieval": mode,
                        "sources": sources, "searched_as": searched}) + "\n\n"
                elif ev["type"] == "token":
                    yield "data: " + json.dumps({"type": "token", "text": ev["text"]}) + "\n\n"
                elif ev["type"] == "done":
                    now = int(time.time())
                    conv["messages"].append({"role": "user",     "content": body.question, "ts": now})
                    conv["messages"].append({"role": "assistant", "content": ev["answer"],
                                             "pages": pages, "retrieval": mode,
                                             "sources": sources, "searched_as": searched, "ts": now})
                    if conv["title"].startswith("Chat ") and len(conv["messages"]) == 2:
                        conv["title"] = body.question[:42]
                    conv["updated_at"] = now
                    save_conv(conv)
                    yield "data: " + json.dumps({
                        "type": "done", "conversation": conv_summary(conv)}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "detail": str(e)[:300]}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

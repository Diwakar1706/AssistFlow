"""
Conversation routes — create, list, get, delete conversations per document.
"""
import time
import uuid

from fastapi import APIRouter, HTTPException

from storage.conversations import (conv_file, conv_summary,
                                   list_convs, load_conv, save_conv)
from storage.documents import load_doc
from api.state import _conv_chats

router = APIRouter()


@router.get("/documents/{doc_id}/conversations")
def list_conversations(doc_id: str):
    load_doc(doc_id)   # 404 if the document is gone
    return [conv_summary(c) for c in list_convs(doc_id)]


@router.post("/documents/{doc_id}/conversations")
def create_conversation(doc_id: str):
    load_doc(doc_id)
    existing = list_convs(doc_id)
    now = int(time.time())
    conv = {
        "id":          uuid.uuid4().hex[:12],
        "document_id": doc_id,
        "title":       f"Chat {len(existing) + 1}",
        "created_at":  now,
        "updated_at":  now,
        "messages":    [],
    }
    save_conv(conv)
    return conv


@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    return load_conv(conv_id)


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    import os
    path = conv_file(conv_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Conversation not found")
    os.remove(path)
    _conv_chats.pop(conv_id, None)
    return {"deleted": conv_id}

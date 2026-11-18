"""
Conversation persistence — one JSON file per conversation in
data/conversations/<conv_id>.json.
"""
import json
import os

from fastapi import HTTPException

from config import CONV_DIR


def conv_file(conv_id: str) -> str:
    return os.path.join(CONV_DIR, f"{conv_id}.json")


def save_conv(conv: dict):
    """Write a conversation record to disk."""
    with open(conv_file(conv["id"]), "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False)


def load_conv(conv_id: str) -> dict:
    """Load a conversation record, or raise HTTP 404."""
    path = conv_file(conv_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Conversation not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def conv_summary(conv: dict) -> dict:
    """Lightweight summary of a conversation (no message payloads)."""
    return {
        "id":            conv["id"],
        "document_id":   conv["document_id"],
        "title":         conv["title"],
        "created_at":    conv["created_at"],
        "updated_at":    conv["updated_at"],
        "message_count": len(conv.get("messages", [])),
    }


def list_convs(doc_id: str) -> list:
    """Return all conversations for a document, newest first."""
    out = []
    for name in os.listdir(CONV_DIR):
        if name.endswith(".json"):
            with open(os.path.join(CONV_DIR, name), encoding="utf-8") as f:
                c = json.load(f)
            if c["document_id"] == doc_id:
                out.append(c)
    out.sort(key=lambda c: c["updated_at"], reverse=True)
    return out

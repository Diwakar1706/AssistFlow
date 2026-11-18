"""
Document metadata persistence — one JSON file per document in data/<doc_id>.json.
"""
import json
import os

from fastapi import HTTPException

from config import DATA_DIR


def doc_file(doc_id: str) -> str:
    return os.path.join(DATA_DIR, f"{doc_id}.json")


def save_doc(doc: dict):
    """Write a document record to disk."""
    with open(doc_file(doc["id"]), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)


def load_doc(doc_id: str) -> dict:
    """Load a document record, or raise HTTP 404."""
    path = doc_file(doc_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Document not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

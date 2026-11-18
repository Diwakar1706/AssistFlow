"""
Document routes — upload, list, get, delete documents.
Handles both PDF file uploads and website URL sources.
"""
import os
import time
import uuid

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from urllib.parse import urlparse

from api.helpers import (assert_public_url, html_to_text,
                          partition_stats, sources_for, text_stats)
from api.models import UrlIn
from config import UPLOAD_DIR, DATA_DIR
from ingestion import describe_pdf_images, get_vector_store, load_and_split_pdf
from storage.documents import doc_file, load_doc, save_doc

router = APIRouter()


# ── Shared ingestion pipeline ──────────────────────────────────────────

def _process_source(doc_id, file_path, display_name, size_bytes,
                    kind="pdf", source_url=None):
    """Partition → Chunk → (Embed if key set). Used by both upload endpoints."""
    steps = {
        "upload":        "done",
        "partitioning":  "done",
        "chunking":      "done",
        "summarisation": "skipped",
        "vectorization": "pending",
    }

    pages, chunks = load_and_split_pdf(file_path)

    if kind == "pdf":
        stats, pages_with_images = partition_stats(file_path)
    else:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            stats = text_stats(f.read(), len(pages))
        pages_with_images = set()

    chunk_list = []
    for i, c in enumerate(chunks):
        base_type = c.metadata.get("type", "text")
        types     = [base_type]
        if base_type == "text" and c.metadata.get("page") in pages_with_images:
            types.append("image")
        chunk_list.append({
            "id":      i + 1,
            "page":    c.metadata.get("page"),
            "chars":   len(c.page_content),
            "content": c.page_content,
            "types":   types,
        })
    avg = int(sum(c["chars"] for c in chunk_list) / len(chunk_list)) if chunk_list else 0

    multimodal = {
        "tables":           sum(1 for c in chunk_list if "table" in c["types"]),
        "images_found":     0,
        "images_described": 0,
        "ocr_pages":        sum(1 for p in pages if p.metadata.get("ocr")),
        "scanned_pages":    sum(1 for p in pages if p.metadata.get("scanned")),
    }

    extra_docs = []
    if kind == "pdf" and os.getenv("OPENAI_API_KEY"):
        try:
            img_docs, found = describe_pdf_images(file_path)
        except Exception:
            img_docs, found = [], 0
        multimodal["images_found"]     = found
        multimodal["images_described"] = len(img_docs)
        extra_docs = img_docs
        for d in img_docs:
            chunk_list.append({
                "id":      len(chunk_list) + 1,
                "page":    d.metadata.get("page"),
                "chars":   len(d.page_content),
                "content": d.page_content,
                "types":   ["image"],
            })

    vector_note = ""
    if os.getenv("OPENAI_API_KEY"):
        try:
            get_vector_store(file_path, extra_chunks=extra_docs or None)
            steps["vectorization"] = "done"
        except Exception as e:
            steps["vectorization"] = "failed"
            vector_note = str(e)[:300]
    else:
        vector_note = ("OPENAI_API_KEY not set in python/.env — chunks are ready, "
                       "add the key and re-upload to enable chat.")

    if multimodal["images_described"] or multimodal["ocr_pages"]:
        steps["summarisation"] = "done"
    if not vector_note and multimodal["scanned_pages"]:
        vector_note = (
            f"{multimodal['scanned_pages']} page(s) look like scanned images and could "
            "not be read (OCR is capped or failed for them) — the rest works normally."
        )

    doc = {
        "id":           doc_id,
        "name":         display_name,
        "kind":         kind,
        "source_url":   source_url,
        "size_bytes":   size_bytes,
        "created_at":   int(time.time()),
        "pdf_path":     file_path,
        "steps":        steps,
        "vector_note":  vector_note,
        "partitioning": stats,
        "multimodal":   multimodal,
        "chunking":     {
            "total_elements":  stats["total_elements"],
            "chunks_created":  len(chunk_list),
            "avg_chunk_chars": avg,
        },
        "chunks": chunk_list,
    }
    save_doc(doc)
    return doc


# ── Endpoints ──────────────────────────────────────────────────────────

@router.get("/documents")
def list_documents():
    docs = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
                import json
                d = json.load(f)
            docs.append({k: d[k] for k in
                         ("id", "name", "size_bytes", "created_at", "steps", "kind", "source_url")
                         if k in d})
    return docs


@router.get("/documents/{doc_id}")
def get_document(doc_id: str):
    return load_doc(doc_id)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file")
    doc_id   = uuid.uuid4().hex[:12]
    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    content  = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)
    return _process_source(doc_id, pdf_path, file.filename, len(content), kind="pdf")


@router.post("/upload-url")
def upload_url(body: UrlIn):
    assert_public_url(body.url)
    try:
        r = requests.get(body.url, timeout=25, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0.0.0 Safari/537.36"})
    except requests.RequestException as e:
        raise HTTPException(400,
            f"Could not fetch that page ({type(e).__name__}). Check the URL and your internet.")
    if r.status_code != 200:
        raise HTTPException(400,
            f"The site returned HTTP {r.status_code} — it may be blocking automated visits or need a login.")
    assert_public_url(r.url)   # re-check after redirects

    ctype  = (r.headers.get("Content-Type") or "").lower()
    doc_id = uuid.uuid4().hex[:12]
    is_pdf = ("application/pdf" in ctype
              or r.content[:4] == b"%PDF"
              or r.url.lower().split("?")[0].endswith(".pdf"))

    if is_pdf:
        path = os.path.join(UPLOAD_DIR, f"{doc_id}_web.pdf")
        with open(path, "wb") as f:
            f.write(r.content)
        name = os.path.basename(urlparse(r.url).path) or "web-document.pdf"
        return _process_source(doc_id, path, name, len(r.content),
                                kind="pdf", source_url=body.url)

    if "text/html" in ctype or "text/plain" in ctype or not ctype:
        if "text/plain" in ctype:
            title, text = urlparse(r.url).hostname or "web page", r.text
        else:
            title, text = html_to_text(r.text)
        if len(text) < 200:
            raise HTTPException(400,
                "Found very little readable text on that page — it is probably built with "
                "JavaScript or behind a login. Try an article/docs page instead.")
        path = os.path.join(UPLOAD_DIR, f"{doc_id}_web.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return _process_source(
            doc_id, path,
            title or (urlparse(r.url).hostname or "web page"),
            len(text.encode("utf-8")),
            kind="url", source_url=body.url)

    raise HTTPException(400,
        f"Unsupported content type ({ctype or 'unknown'}) — only web pages and PDF links are supported.")


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Remove a document completely: file, chunk data, ChromaDB store, and all its chats."""
    import shutil
    from storage.conversations import conv_file, list_convs
    from api.state import _chats, _conv_chats

    doc = load_doc(doc_id)
    for conv in list_convs(doc_id):
        _conv_chats.pop(conv["id"], None)
        p = conv_file(conv["id"])
        if os.path.exists(p):
            os.remove(p)
    _chats.pop(doc_id, None)

    pdf_path = doc.get("pdf_path", "")
    if pdf_path and os.path.exists(pdf_path):
        os.remove(pdf_path)

    try:
        from ingestion.vectorstore import pdf_id
        from config import DB_ROOT
        store_dir = os.path.join(DB_ROOT, pdf_id(pdf_path))
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir, ignore_errors=True)
    except Exception:
        pass

    os.remove(doc_file(doc_id))
    return {"deleted": doc_id}

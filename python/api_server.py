"""
FastAPI backend for the React frontend (frontend/ folder).

What it does when you upload a PDF — the same pipeline the UI shows, step by step:
  1. Upload                  -> file is saved locally in python/uploads/   (NO S3, NO queue — everything is local & instant)
  2. Partitioning            -> pypdf reads the PDF page by page and counts elements (text sections, titles/headers, images)
  3. Chunking                -> RecursiveCharacterTextSplitter (1000 chars, 200 overlap), every chunk keeps its page number
  4. Summarisation           -> SKIPPED in this fast local mode (only needed for image/table chunks — see lesson 08)
  5. Vectorization & Storage -> OpenAI embeddings -> ChromaDB on disk (needs OPENAI_API_KEY in python/.env)
  6. View Chunks / Chat      -> chunks are returned to the UI; chat answers only from this PDF, with page citations

Run:
  cd python
  pip install -r requirements.txt        (includes fastapi + uvicorn)
  uvicorn api_server:app --reload --port 8000
"""
import json
import os
import shutil
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from rag_core import PdfChat, describe_pdf_images, get_vector_store, load_and_split_pdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="RAG for Beginners — local API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_chats = {}  # doc_id -> PdfChat (keeps short history per document)


def _partition_stats(pdf_path):
    """Count elements in the PDF. Text stats via pypdf; tables detected with
    PyMuPDF (find_tables) and images counted per page.
    Titles/headers = heuristic: short lines that don't end with a full stop."""
    reader = PdfReader(pdf_path)
    text_sections, titles, images = 0, 0, 0
    pages_with_images = set()
    for pno, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for block in text.split("\n\n"):
            if block.strip():
                text_sections += 1
        for line in text.splitlines():
            s = line.strip()
            if 3 < len(s) < 80 and not s.endswith(".") and len(s.split()) <= 10:
                titles += 1
        try:
            n_img = len(page.images)
        except Exception:
            n_img = 0
        if n_img:
            pages_with_images.add(pno)
            images += n_img
    tables = 0
    try:
        import fitz
        fdoc = fitz.open(pdf_path)
        for page in fdoc:
            try:
                tables += len(page.find_tables().tables)
            except Exception:
                pass
        fdoc.close()
    except Exception:
        tables = 0
    total = text_sections + titles + images + tables
    return {
        "pages": len(reader.pages),
        "text_sections": text_sections,
        "titles_headers": titles,
        "images": images,
        "tables": tables,
        "other_elements": max(total - text_sections - titles - images, 0),
        "total_elements": total,
    }, pages_with_images


def _doc_file(doc_id):
    return os.path.join(DATA_DIR, f"{doc_id}.json")


def _save_doc(doc):
    with open(_doc_file(doc["id"]), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)


def _load_doc(doc_id):
    path = _doc_file(doc_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Document not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {"ok": True, "openai_key_set": bool(os.getenv("OPENAI_API_KEY")), "cohere_key_set": bool(os.getenv("COHERE_API_KEY"))}


@app.get("/")
def root():
    """Friendly landing so opening http://localhost:8000 doesn't look like an error."""
    return {
        "service": "RAG for Beginners — local API",
        "status": "running",
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "frontend": "Open the React UI at http://localhost:5173 (or the port Vite printed)",
        "endpoints": ["/api/health", "/api/documents", "/api/upload", "/api/chat"],
    }


@app.get("/health")
def health_alias():
    return health()


@app.get("/api/documents")
def list_documents():
    docs = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
                d = json.load(f)
            docs.append({k: d[k] for k in ("id", "name", "size_bytes", "created_at", "steps", "kind", "source_url") if k in d})
    return docs


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    return _load_doc(doc_id)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file")
    doc_id = uuid.uuid4().hex[:12]
    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    content = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)
    return _process_source(doc_id, pdf_path, file.filename, len(content), kind="pdf")


def _text_stats(text, num_segments):
    """Partitioning-style stats for a web/text source (same keys as the PDF stats)."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    titles = sum(1 for line in text.splitlines()
                 if 3 < len(line.strip()) < 80 and not line.strip().endswith(".")
                 and len(line.split()) <= 10)
    return {"text_sections": num_segments, "titles_headers": titles, "images": 0,
            "tables": 0, "total_elements": len(paragraphs) + titles}


def _process_source(doc_id, file_path, display_name, size_bytes, kind="pdf", source_url=None):
    """The shared pipeline for every source type: partition -> chunk -> (embed if key set)."""
    steps = {"upload": "done", "partitioning": "done", "chunking": "done",
             "summarisation": "skipped", "vectorization": "pending"}

    # 3) Chunking first (keeps 1-based page/section numbers)
    pages, chunks = load_and_split_pdf(file_path)
    # 2) Partitioning stats (PDF: pypdf element counts; web: text-based counts)
    if kind == "pdf":
        stats, pages_with_images = _partition_stats(file_path)
    else:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            stats = _text_stats(f.read(), len(pages))
        pages_with_images = set()
    chunk_list = []
    for i, c in enumerate(chunks):
        base_type = c.metadata.get("type", "text")
        types = [base_type]
        if base_type == "text" and c.metadata.get("page") in pages_with_images:
            types.append("image")  # this text chunk's page also contains an image
        chunk_list.append({
            "id": i + 1,
            "page": c.metadata.get("page"),
            "chars": len(c.page_content),
            "content": c.page_content,
            "types": types,
        })
    avg = int(sum(c["chars"] for c in chunk_list) / len(chunk_list)) if chunk_list else 0

    # Multi-modal extras: GPT-4o descriptions of embedded images become searchable chunks
    multimodal = {
        "tables": sum(1 for c in chunk_list if "table" in c["types"]),
        "images_found": 0,
        "images_described": 0,
        "ocr_pages": sum(1 for p in pages if p.metadata.get("ocr")),
        "scanned_pages": sum(1 for p in pages if p.metadata.get("scanned")),
    }
    extra_docs = []
    if kind == "pdf" and os.getenv("OPENAI_API_KEY"):
        try:
            img_docs, found = describe_pdf_images(file_path)
        except Exception:
            img_docs, found = [], 0
        multimodal["images_found"] = found
        multimodal["images_described"] = len(img_docs)
        extra_docs = img_docs
        for d in img_docs:
            chunk_list.append({
                "id": len(chunk_list) + 1,
                "page": d.metadata.get("page"),
                "chars": len(d.page_content),
                "content": d.page_content,
                "types": ["image"],
            })

    # 5) Vectorization & Storage — only possible with an OpenAI key
    vector_note = ""
    if os.getenv("OPENAI_API_KEY"):
        try:
            get_vector_store(file_path, extra_chunks=extra_docs or None)
            steps["vectorization"] = "done"
        except Exception as e:  # keep the doc usable for viewing chunks even if embedding fails
            steps["vectorization"] = "failed"
            vector_note = str(e)[:300]
    else:
        vector_note = "OPENAI_API_KEY not set in python/.env — chunks are ready, add the key and re-upload to enable chat."
    if multimodal["images_described"] or multimodal["ocr_pages"]:
        steps["summarisation"] = "done"
    if not vector_note and multimodal["scanned_pages"]:
        vector_note = (f"{multimodal['scanned_pages']} page(s) look like scanned images and could not be read "
                       "(OCR is capped or failed for them) — the rest of the document works normally.")

    doc = {
        "id": doc_id,
        "name": display_name,
        "kind": kind,
        "source_url": source_url,
        "size_bytes": size_bytes,
        "created_at": int(time.time()),
        "pdf_path": file_path,
        "steps": steps,
        "vector_note": vector_note,
        "partitioning": stats,
        "multimodal": multimodal,
        "chunking": {"total_elements": stats["total_elements"], "chunks_created": len(chunk_list), "avg_chunk_chars": avg},
        "chunks": chunk_list,
    }
    _save_doc(doc)
    return doc


# ──────────────────────────────────────────────────────────────────────
# Add a source from a website URL: fetch the page (or a PDF link),
# extract the readable text, then run the exact same pipeline as an upload.
# ──────────────────────────────────────────────────────────────────────
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests


class UrlIn(BaseModel):
    url: str


def _assert_public_url(url):
    """Basic safety for URL fetching: public http(s) pages only. Blocks localhost
    and literal private/loopback IPs typed into the box. (No DNS-resolution check:
    this runs on the user's own machine for a URL they pasted themselves, and
    proxy/VPN/NAT64 DNS answers would cause false blocks on real public sites.)"""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "Please paste a valid http:// or https:// URL")
    host = p.hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        raise HTTPException(400, "Local addresses are not allowed — paste a public web page.")
    try:
        ip = ipaddress.ip_address(host)  # only when the host IS an IP literal
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved):
        raise HTTPException(400, "Local/private addresses are not allowed — paste a public web page.")


class _ReadableText(HTMLParser):
    """Minimal readable-text extractor (no extra dependency): skips scripts,
    styles, nav/header/footer, and turns block elements into paragraphs."""
    SKIP = {"script", "style", "nav", "header", "footer", "aside", "form", "button", "svg", "noscript"}
    BLOCK = {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._skip = [], "", 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _html_to_text(html):
    parser = _ReadableText()
    parser.feed(html)
    raw = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    text = "\n\n".join(p for p in (line.strip() for line in lines) if p)
    return parser.title.strip(), text


@app.post("/api/upload-url")
def upload_url(body: UrlIn):
    _assert_public_url(body.url)
    try:
        r = requests.get(body.url, timeout=25, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"})
    except requests.RequestException as e:
        raise HTTPException(400, f"Could not fetch that page ({type(e).__name__}). Check the URL and your internet.")
    if r.status_code != 200:
        raise HTTPException(400, f"The site returned HTTP {r.status_code} — it may be blocking automated visits or need a login.")
    _assert_public_url(r.url)  # re-check after redirects

    ctype = (r.headers.get("Content-Type") or "").lower()
    doc_id = uuid.uuid4().hex[:12]
    is_pdf = "application/pdf" in ctype or r.content[:4] == b"%PDF" or r.url.lower().split("?")[0].endswith(".pdf")
    if is_pdf:
        path = os.path.join(UPLOAD_DIR, f"{doc_id}_web.pdf")
        with open(path, "wb") as f:
            f.write(r.content)
        name = os.path.basename(urlparse(r.url).path) or "web-document.pdf"
        return _process_source(doc_id, path, name, len(r.content), kind="pdf", source_url=body.url)
    if "text/html" in ctype or "text/plain" in ctype or not ctype:
        if "text/plain" in ctype:
            title, text = urlparse(r.url).hostname or "web page", r.text
        else:
            title, text = _html_to_text(r.text)
        if len(text) < 200:
            raise HTTPException(400, "Found very little readable text on that page — it is probably built "
                                     "with JavaScript or behind a login. Try an article/docs page instead.")
        path = os.path.join(UPLOAD_DIR, f"{doc_id}_web.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return _process_source(doc_id, path, title or (urlparse(r.url).hostname or "web page"),
                               len(text.encode("utf-8")), kind="url", source_url=body.url)
    raise HTTPException(400, f"Unsupported content type ({ctype or 'unknown'}) — only web pages and PDF links are supported.")


def _sources_for(doc, docs):
    """Map the chunks an answer was built from to View-Chunks entries
    (chunk_id matches the numbering in the pipeline's View Chunks tab)."""
    by_content = {}
    for c in doc.get("chunks", []):
        by_content.setdefault(c["content"], c["id"])
    out, seen = [], set()
    for d in docs:
        cid = d.metadata.get("chunk_id") or by_content.get(d.page_content)
        if cid is None:  # last resort: same page + overlapping text
            for c in doc.get("chunks", []):
                if c["page"] == d.metadata.get("page") and (
                        c["content"][:80] in d.page_content or d.page_content[:80] in c["content"]):
                    cid = c["id"]
                    break
        if cid is not None and cid not in seen:
            seen.add(cid)
            out.append({"chunk_id": cid, "page": d.metadata.get("page"),
                        "preview": d.page_content[:220]})
    return out


class ChatIn(BaseModel):
    document_id: str
    question: str


@app.post("/api/chat")
def chat(body: ChatIn):
    doc = _load_doc(body.document_id)
    if doc["steps"].get("vectorization") != "done":
        raise HTTPException(400, doc.get("vector_note") or "This document is not vectorized yet.")
    if body.document_id not in _chats:
        store, _, _, _ = get_vector_store(doc["pdf_path"])
        from langchain_core.documents import Document
        chunks_as_docs = [
            Document(page_content=c["content"], metadata={"page": c["page"], "source_file": doc["name"], "chunk_id": c["id"]})
            for c in doc.get("chunks", [])
        ]
        _chats[body.document_id] = PdfChat(store, docs=chunks_as_docs)
    chat_obj = _chats[body.document_id]
    answer, pages, used_docs = chat_obj.ask(body.question)
    searched = chat_obj.last_query_used if chat_obj.last_query_used and chat_obj.last_query_used != body.question else None
    return {"answer": answer, "pages": pages, "retrieval": chat_obj.last_mode,
            "sources": _sources_for(doc, used_docs), "searched_as": searched}


# ──────────────────────────────────────────────────────────────────────
# Conversations — every PDF has its OWN chat history.
# One document -> many conversations -> each conversation keeps its messages.
# Stored as JSON files in python/data/conversations/ so history survives restarts.
# ──────────────────────────────────────────────────────────────────────
CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

_conv_chats = {}  # conversation_id -> PdfChat (seeded from that conversation's messages)


def _conv_file(conv_id):
    return os.path.join(CONV_DIR, f"{conv_id}.json")


def _save_conv(conv):
    with open(_conv_file(conv["id"]), "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False)


def _load_conv(conv_id):
    path = _conv_file(conv_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Conversation not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _conv_summary(conv):
    return {
        "id": conv["id"],
        "document_id": conv["document_id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "message_count": len(conv.get("messages", [])),
    }


def _list_convs(doc_id):
    out = []
    for name in os.listdir(CONV_DIR):
        if name.endswith(".json"):
            with open(os.path.join(CONV_DIR, name), encoding="utf-8") as f:
                c = json.load(f)
            if c["document_id"] == doc_id:
                out.append(c)
    out.sort(key=lambda c: c["updated_at"], reverse=True)
    return out


@app.get("/api/documents/{doc_id}/conversations")
def list_conversations(doc_id: str):
    _load_doc(doc_id)  # 404 if the document is gone
    return [_conv_summary(c) for c in _list_convs(doc_id)]


@app.post("/api/documents/{doc_id}/conversations")
def create_conversation(doc_id: str):
    _load_doc(doc_id)
    existing = _list_convs(doc_id)
    now = int(time.time())
    conv = {
        "id": uuid.uuid4().hex[:12],
        "document_id": doc_id,
        "title": f"Chat {len(existing) + 1}",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _save_conv(conv)
    return conv


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str):
    return _load_conv(conv_id)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    path = _conv_file(conv_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Conversation not found")
    os.remove(path)
    _conv_chats.pop(conv_id, None)
    return {"deleted": conv_id}


def _get_conv_chat(conv):
    """PdfChat for one conversation, its memory seeded from the saved messages
    (last 3 turns) so follow-up questions keep working after a server restart."""
    if conv["id"] in _conv_chats:
        return _conv_chats[conv["id"]]
    doc = _load_doc(conv["document_id"])
    store, _, _, _ = get_vector_store(doc["pdf_path"])
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage as _AI, HumanMessage as _HU
    chunks_as_docs = [
        Document(page_content=c["content"], metadata={"page": c["page"], "source_file": doc["name"], "chunk_id": c["id"]})
        for c in doc.get("chunks", [])
    ]
    chat_obj = PdfChat(store, docs=chunks_as_docs)
    history = []
    for m in conv.get("messages", [])[-6:]:
        history.append(_HU(content=m["content"]) if m["role"] == "user" else _AI(content=m["content"]))
    chat_obj.history = history
    _conv_chats[conv["id"]] = chat_obj
    return chat_obj


class ConvChatIn(BaseModel):
    question: str


@app.post("/api/conversations/{conv_id}/chat")
def chat_in_conversation(conv_id: str, body: ConvChatIn):
    conv = _load_conv(conv_id)
    doc = _load_doc(conv["document_id"])
    if doc["steps"].get("vectorization") != "done":
        raise HTTPException(400, doc.get("vector_note") or "This document is not vectorized yet.")
    chat_obj = _get_conv_chat(conv)
    answer, pages, used_docs = chat_obj.ask(body.question)
    sources = _sources_for(doc, used_docs)
    searched = chat_obj.last_query_used if chat_obj.last_query_used and chat_obj.last_query_used != body.question else None
    now = int(time.time())
    conv["messages"].append({"role": "user", "content": body.question, "ts": now})
    conv["messages"].append({"role": "assistant", "content": answer, "pages": pages,
                              "retrieval": chat_obj.last_mode, "sources": sources,
                              "searched_as": searched, "ts": now})
    if conv["title"].startswith("Chat ") and len(conv["messages"]) == 2:
        conv["title"] = body.question[:42]  # first question becomes the chat's title
    conv["updated_at"] = now
    _save_conv(conv)
    return {"answer": answer, "pages": pages, "retrieval": chat_obj.last_mode,
            "sources": sources, "searched_as": searched,
            "conversation": _conv_summary(conv)}


@app.post("/api/conversations/{conv_id}/chat/stream")
def chat_in_conversation_stream(conv_id: str, body: ConvChatIn):
    """Streaming version of the chat endpoint (Server-Sent Events).
    Events: meta (pages/sources/retrieval/searched_as) -> token (many) -> done | error.
    The finished answer is saved to the conversation history exactly like the normal endpoint."""
    conv = _load_conv(conv_id)
    doc = _load_doc(conv["document_id"])
    if doc["steps"].get("vectorization") != "done":
        raise HTTPException(400, doc.get("vector_note") or "This document is not vectorized yet.")
    chat_obj = _get_conv_chat(conv)

    def event_stream():
        sources, searched, mode, pages = [], None, "", []
        try:
            for ev in chat_obj.ask_stream(body.question):
                if ev["type"] == "context":
                    pages, mode, searched = ev["pages"], ev["mode"], ev["searched_as"]
                    sources = _sources_for(doc, ev["docs"])
                    yield "data: " + json.dumps({"type": "meta", "pages": pages, "retrieval": mode,
                                                 "sources": sources, "searched_as": searched}) + "\n\n"
                elif ev["type"] == "token":
                    yield "data: " + json.dumps({"type": "token", "text": ev["text"]}) + "\n\n"
                elif ev["type"] == "done":
                    now = int(time.time())
                    conv["messages"].append({"role": "user", "content": body.question, "ts": now})
                    conv["messages"].append({"role": "assistant", "content": ev["answer"], "pages": pages,
                                             "retrieval": mode, "sources": sources,
                                             "searched_as": searched, "ts": now})
                    if conv["title"].startswith("Chat ") and len(conv["messages"]) == 2:
                        conv["title"] = body.question[:42]
                    conv["updated_at"] = now
                    _save_conv(conv)
                    yield "data: " + json.dumps({"type": "done",
                                                 "conversation": _conv_summary(conv)}) + "\n\n"
        except Exception as e:  # keep the stream well-formed even on failure
            yield "data: " + json.dumps({"type": "error", "detail": str(e)[:300]}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Remove a PDF completely: uploaded file, chunk data, ChromaDB store, and all its chats."""
    doc = _load_doc(doc_id)
    # conversations + their cached chats
    for conv in _list_convs(doc_id):
        _conv_chats.pop(conv["id"], None)
        p = _conv_file(conv["id"])
        if os.path.exists(p):
            os.remove(p)
    _chats.pop(doc_id, None)
    # uploaded pdf file
    pdf_path = doc.get("pdf_path", "")
    if pdf_path and os.path.exists(pdf_path):
        os.remove(pdf_path)
    # chroma vector store for this pdf
    try:
        from rag_core import DB_ROOT, _pdf_id
        store_dir = os.path.join(DB_ROOT, _pdf_id(pdf_path))
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir, ignore_errors=True)
    except Exception:
        pass
    # document record
    os.remove(_doc_file(doc_id))
    return {"deleted": doc_id}

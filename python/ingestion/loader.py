"""
Document loading — PDF pages, web-page text segments, OCR for scanned
pages, and GPT-4o Vision descriptions of embedded images.
"""
import base64
import os

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import VISION_MODEL, IMAGE_MAX_DESCRIBE, MIN_IMAGE_PX

try:
    import fitz          # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz   # type: ignore
    except ImportError:
        fitz = None


# ── Plain-text / web source ────────────────────────────────────────────

def load_text_segments(txt_path: str):
    """Load a plain-text source (e.g. text extracted from a web page) as
    page-like segments of ~2500 chars, split at paragraph boundaries.
    Each segment gets a 0-based 'page' number so the rest of the pipeline
    (chunking, citations, sources) works unchanged."""
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    segments, buf = [], ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) > 2500:
            segments.append(buf)
            buf = para
        else:
            buf = (buf + "\n\n" + para) if buf else para
    if buf:
        segments.append(buf)
    return [Document(page_content=s, metadata={"page": i, "source": txt_path})
            for i, s in enumerate(segments)]


# ── Vision helpers ─────────────────────────────────────────────────────

def _vision_message(prompt: str, png_bytes: bytes):
    """Build a multi-modal HumanMessage with an inline base64 image."""
    b64 = base64.b64encode(png_bytes).decode()
    return [HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])]


def ocr_page_text(fitz_page) -> str:
    """OCR one scanned page with GPT-4o Vision (no Tesseract needed)."""
    pix = fitz_page.get_pixmap(dpi=200)
    res = ChatOpenAI(model=VISION_MODEL, temperature=0).invoke(_vision_message(
        "Transcribe ALL text visible in this scanned document page exactly, "
        "in reading order. Output only the transcription, no commentary.",
        pix.tobytes("png")))
    return (res.content if isinstance(res.content, str) else str(res.content)).strip()


# ── Table extraction ───────────────────────────────────────────────────

def extract_table_docs(fitz_doc, source_name: str):
    """Pull tables out as structured chunks ('cell | cell | cell') so
    questions about premium / price / benefit tables can be answered."""
    out = []
    for pno, page in enumerate(fitz_doc):
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []
        for t in tables:
            try:
                rows = t.extract()
            except Exception:
                continue
            lines = []
            for row in rows:
                cells = [(c or "").strip() for c in row]
                if any(cells):
                    lines.append(" | ".join(cells))
            if len(lines) >= 2:
                out.append(Document(
                    page_content=f"[Table on page {pno + 1}]\n" + "\n".join(lines),
                    metadata={"page": pno, "source_file": source_name, "type": "table"}))
    return out


# ── Image description ──────────────────────────────────────────────────

def describe_pdf_images(pdf_path: str, max_images: int = None):
    """Extract embedded images and describe each with GPT-4o Vision,
    producing searchable text chunks.
    Returns (image_chunk_documents, total_images_found)."""
    if fitz is None or not os.getenv("OPENAI_API_KEY"):
        return [], 0
    max_images = IMAGE_MAX_DESCRIBE if max_images is None else max_images
    docs, found, seen = [], 0, set()
    try:
        fdoc = fitz.open(pdf_path)
    except Exception:
        return [], 0
    llm = None
    for pno, page in enumerate(fdoc):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                pix = fitz.Pixmap(fdoc, xref)
                if pix.width < MIN_IMAGE_PX or pix.height < MIN_IMAGE_PX:
                    continue   # tiny icon / logo — not worth a vision call
                found += 1
                if len(docs) >= max_images:
                    continue
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if llm is None:
                    llm = ChatOpenAI(model=VISION_MODEL, temperature=0)
                res = llm.invoke(_vision_message(
                    "Describe this image from a document in detail for search: what it shows, "
                    "its type (chart/photo/diagram/table), every number, label and piece of "
                    "text visible, and the main trend or message. Plain text only.",
                    pix.tobytes("png")))
                desc = (res.content if isinstance(res.content, str) else str(res.content)).strip()
                if desc:
                    docs.append(Document(
                        page_content=f"[Image on page {pno + 1}] {desc}",
                        metadata={"page": pno + 1,
                                  "source_file": os.path.basename(pdf_path),
                                  "type": "image"}))
            except Exception:
                continue
    fdoc.close()
    return docs, found

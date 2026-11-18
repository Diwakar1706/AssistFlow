"""
Document chunking — split loaded pages into retrieval-ready chunks,
including structured table chunks and OCR for scanned pages.
"""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import OCR_MAX_PAGES
from ingestion.loader import load_text_segments, extract_table_docs, ocr_page_text

try:
    import fitz
except ImportError:
    try:
        import pymupdf as fitz   # type: ignore
    except ImportError:
        fitz = None


def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load a PDF (or a .txt web source) and split into chunks that keep
    their page number.  PDFs also get:
      - Structured TABLE chunks (PyMuPDF)
      - OCR for scanned pages (GPT-4o Vision, capped by OCR_MAX_PAGES)
    Returns (pages, chunks).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if pdf_path.lower().endswith(".txt"):
        pages      = load_text_segments(pdf_path)
        table_docs = []
    else:
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()          # one Document per page (0-based 'page' in metadata)
        table_docs = []
        if fitz is not None:
            try:
                fdoc    = fitz.open(pdf_path)
                has_key = bool(os.getenv("OPENAI_API_KEY"))
                ocr_done = 0
                for i, pdoc in enumerate(pages):
                    if len(pdoc.page_content.strip()) >= 20 or i >= len(fdoc):
                        continue   # page has readable text — no OCR needed
                    if has_key and ocr_done < OCR_MAX_PAGES:
                        try:
                            text = ocr_page_text(fdoc[i])
                        except Exception:
                            text = ""
                        if text:
                            pdoc.page_content    = text
                            pdoc.metadata["ocr"] = True
                            ocr_done += 1
                            continue
                    pdoc.metadata["scanned"] = True
                table_docs = extract_table_docs(fdoc, os.path.basename(pdf_path))
                fdoc.close()
            except Exception:
                table_docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages + table_docs)

    # Normalise metadata: 1-based page numbers, source_file, type
    for c in chunks:
        c.metadata["page"]        = int(c.metadata.get("page", 0)) + 1
        c.metadata["source_file"] = os.path.basename(pdf_path)
        c.metadata.setdefault("type", "text")

    return pages, chunks

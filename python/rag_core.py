"""
Shared RAG core for the PDF Q&A app.
Idea: a PDF can be 100+ pages (like an insurance / company policy).
You ask a question in normal human language, the answer comes ONLY from that PDF.

Flow (same flow the lessons in this folder teach, combined into one pipeline):
  1. Load PDF page-by-page          -> every chunk keeps its page number
  2. Split into small chunks         -> RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
  3. Embed chunks with OpenAI        -> text-embedding-3-small
  4. Store in ChromaDB (on disk)     -> saved per-PDF, so a 2nd run does NOT pay to re-embed
  5. On a question: retrieve top chunks -> send only those chunks to GPT -> answer + page citations
"""
import hashlib
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hybrid search (lesson 12): vector (semantic) + BM25 (keyword) combined
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # newer langchain layouts
    from langchain_community.retrievers import EnsembleRetriever  # type: ignore
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.retrievers import BM25Retriever  # type: ignore

try:
    import fitz  # PyMuPDF — tables, images, page rendering (pure pip install, works on Windows)
except ImportError:
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        fitz = None

VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")   # vision model for image descriptions + OCR
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "30"))       # cost cap: scanned pages OCR'd per PDF
IMAGE_MAX_DESCRIBE = int(os.getenv("IMAGE_MAX_DESCRIBE", "15"))  # cost cap: images described per PDF
MIN_IMAGE_PX = 120                                          # ignore tiny icons/logos

load_dotenv()

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # set OPENAI_MODEL=gpt-4o in .env for the stronger model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_ROOT = os.path.join(BASE_DIR, "db")


def _pdf_id(pdf_path: str) -> str:
    """Stable id for a PDF file (name + size + first bytes) -> its own vector-store folder."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read(1024 * 1024))
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40]
    return f"{stem}_{h.hexdigest()[:8]}"


def _load_text_segments(txt_path: str):
    """Load a plain-text source (e.g. text extracted from a web page) as page-like
    segments of ~2500 chars, split at paragraph boundaries. Each segment gets a
    0-based 'page' number, exactly like a PDF page, so the rest of the pipeline
    (chunking, citations, sources) works unchanged — 'page' means 'section' here."""
    from langchain_core.documents import Document
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


def _vision_message(prompt, png_bytes):
    import base64
    b64 = base64.b64encode(png_bytes).decode()
    return [HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])]


def _ocr_page_text(fitz_page) -> str:
    """OCR one scanned page with GPT-4o vision (no Tesseract install needed)."""
    pix = fitz_page.get_pixmap(dpi=200)
    res = ChatOpenAI(model=VISION_MODEL, temperature=0).invoke(_vision_message(
        "Transcribe ALL text visible in this scanned document page exactly, in reading order. "
        "Output only the transcription, no commentary.", pix.tobytes("png")))
    return (res.content if isinstance(res.content, str) else str(res.content)).strip()


def _extract_table_docs(fitz_doc, source_name):
    """Pull tables out as structured chunks ('cell | cell | cell') so questions
    about premium/price/benefit tables can actually be answered."""
    from langchain_core.documents import Document
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


def describe_pdf_images(pdf_path, max_images=None):
    """Extract embedded images and describe each with GPT-4o vision, producing
    searchable chunks (lesson 08's idea, without the heavy Unstructured stack).
    Returns (image_chunk_documents, total_images_found)."""
    from langchain_core.documents import Document
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
                    continue  # tiny icon / logo — not worth a vision call
                found += 1
                if len(docs) >= max_images:
                    continue
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if llm is None:
                    llm = ChatOpenAI(model=VISION_MODEL, temperature=0)
                res = llm.invoke(_vision_message(
                    "Describe this image from a document in detail for search: what it shows, its type "
                    "(chart/photo/diagram/table), every number, label and piece of text visible, and the "
                    "main trend or message. Plain text only.", pix.tobytes("png")))
                desc = (res.content if isinstance(res.content, str) else str(res.content)).strip()
                if desc:
                    docs.append(Document(
                        page_content=f"[Image on page {pno + 1}] {desc}",
                        metadata={"page": pno + 1, "source_file": os.path.basename(pdf_path), "type": "image"}))
            except Exception:
                continue
    fdoc.close()
    return docs, found


def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load a PDF (or a .txt web source) and split it into chunks that remember their page number.
    PDFs also get: structured TABLE chunks (PyMuPDF) and OCR for scanned pages (GPT-4o vision,
    only when an OpenAI key is set, capped by OCR_MAX_PAGES). Page docs carry metadata flags
    'ocr' / 'scanned' so callers can report what happened."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.lower().endswith(".txt"):
        pages = _load_text_segments(pdf_path)
        table_docs = []
    else:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # one Document per page, metadata has 'page' (0-based) and 'source'
        table_docs = []
        if fitz is not None:
            try:
                fdoc = fitz.open(pdf_path)
                has_key = bool(os.getenv("OPENAI_API_KEY"))
                ocr_done = 0
                for i, pdoc in enumerate(pages):
                    if len(pdoc.page_content.strip()) >= 20 or i >= len(fdoc):
                        continue
                    # a page with (almost) no text layer = scanned image page
                    if has_key and ocr_done < OCR_MAX_PAGES:
                        try:
                            text = _ocr_page_text(fdoc[i])
                        except Exception:
                            text = ""
                        if text:
                            pdoc.page_content = text
                            pdoc.metadata["ocr"] = True
                            ocr_done += 1
                            continue
                    pdoc.metadata["scanned"] = True
                table_docs = _extract_table_docs(fdoc, os.path.basename(pdf_path))
                fdoc.close()
            except Exception:
                table_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages + table_docs)
    for c in chunks:  # make page numbers human (1-based) once, here
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1
        c.metadata["source_file"] = os.path.basename(pdf_path)
        c.metadata.setdefault("type", "text")
    return pages, chunks


def get_vector_store(pdf_path: str, force_rebuild: bool = False, extra_chunks=None):
    """Create (or reuse) the ChromaDB store for this PDF. Returns (vectorstore, num_pages, num_chunks, reused).
    extra_chunks (e.g. GPT-4o image-description chunks) are embedded too; when they are
    given, the store is rebuilt so the extras are definitely included."""
    persist_dir = os.path.join(DB_ROOT, _pdf_id(pdf_path))
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if extra_chunks:
        force_rebuild = True

    if os.path.exists(persist_dir) and not force_rebuild:
        store = Chroma(persist_directory=persist_dir, embedding_function=embeddings,
                       collection_metadata={"hnsw:space": "cosine"})
        count = store._collection.count()
        if count > 0:
            return store, None, count, True

    pages, chunks = load_and_split_pdf(pdf_path)
    if extra_chunks:
        chunks = chunks + list(extra_chunks)
    store = None
    batch = 100  # add in batches so very large PDFs (100+ pages) don't hit API limits in one call
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        if store is None:
            store = Chroma.from_documents(documents=part, embedding=embeddings,
                                          persist_directory=persist_dir,
                                          collection_metadata={"hnsw:space": "cosine"})
        else:
            store.add_documents(part)
        print(f"  embedded chunks {min(i + batch, len(chunks))}/{len(chunks)}")
    return store, len(pages), len(chunks), False


class PdfChat:
    """Ask plain-English questions about one indexed PDF. Keeps short chat history.

    Retrieval pipeline (upgraded — lessons 12 + 13 integrated):
      1. HYBRID SEARCH: vector similarity (Chroma) + BM25 keyword search, fused by
         EnsembleRetriever (weights 0.7 / 0.3). BM25 catches exact terms, clause
         numbers and amounts that pure vector search can miss.
      2. RERANK: the top candidates are re-ordered — Cohere Rerank if COHERE_API_KEY
         is set (lesson 13), otherwise a GPT-4o-mini listwise rerank using the
         OpenAI key you already have. Any rerank failure falls back to hybrid order.
      3. ANSWER: only the final top-k chunks go to the chat model.
    """

    def __init__(self, vectorstore, docs=None, model: str = CHAT_MODEL, k: int = 6, candidate_k: int = 12):
        self.db = vectorstore
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.k = k
        self.candidate_k = candidate_k
        self.history = []
        self.last_mode = "vector"
        self.last_query_used = None
        self._hybrid = None
        if docs:
            try:
                bm25 = BM25Retriever.from_documents(docs)
                bm25.k = candidate_k
                vector = vectorstore.as_retriever(search_kwargs={"k": candidate_k})
                self._hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.7, 0.3])
            except Exception:
                self._hybrid = None

    # -- step 1: hybrid retrieval -------------------------------------------
    def _retrieve_candidates(self, question):
        if self._hybrid is not None:
            try:
                docs = self._hybrid.invoke(question)
                self.last_mode = "hybrid (vector + BM25)"
                return docs
            except Exception:
                pass
        retriever = self.db.as_retriever(search_kwargs={"k": self.candidate_k})
        self.last_mode = "vector only"
        return retriever.invoke(question)

    # -- step 2: rerank ------------------------------------------------------
    def _rerank(self, question, candidates):
        if len(candidates) <= self.k:
            return candidates
        # 2a. Cohere Rerank (lesson 13) — only if the user added a Cohere key
        if os.getenv("COHERE_API_KEY"):
            try:
                from langchain_cohere import CohereRerank
                reranker = CohereRerank(model="rerank-english-v3.0", top_n=self.k)
                out = reranker.compress_documents(candidates, question)
                if out:
                    self.last_mode += " + Cohere rerank"
                    return out
            except Exception:
                pass
        # 2b. LLM listwise rerank with the OpenAI model (no extra key needed)
        try:
            listing = "\n\n".join(
                f"[{i}] (Page {d.metadata.get('page', '?')}) {d.page_content[:250]}"
                for i, d in enumerate(candidates)
            )
            prompt = (
                f"Question: {question}\n\nPassages:\n{listing}\n\n"
                f"Return ONLY a JSON list of the passage numbers of the {self.k} most relevant "
                "passages, most relevant first, e.g. [3, 0, 5]. No other text."
            )
            res = self.llm.invoke([HumanMessage(content=prompt)])
            text = res.content if isinstance(res.content, str) else str(res.content)
            import json as _json
            import re as _re
            m = _re.search(r"\[[0-9,\s]+\]", text)
            order = _json.loads(m.group(0)) if m else []
            picked, seen = [], set()
            for idx in order:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    picked.append(candidates[idx])
                    seen.add(idx)
                if len(picked) >= self.k:
                    break
            if picked:
                self.last_mode += " + LLM rerank"
                return picked
        except Exception:
            pass
        return candidates[: self.k]

    def retrieve(self, question):
        candidates = self._retrieve_candidates(question)
        return self._rerank(question, candidates)

    # -- step 0: follow-up aware search --------------------------------------
    def _condense_question(self, question):
        """Rewrite a follow-up into a STANDALONE question using the chat history,
        so retrieval searches for what the user actually means:
            "and what about its cost?"  ->  "What is the premium for the family floater plan?"
        No history, already-standalone, or any failure -> returns the question unchanged."""
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
                "Rewrite the follow-up as a complete standalone question containing every detail "
                "needed to search a document for its answer (names, plan, topic from the conversation). "
                "If it is already standalone, return it unchanged. Return ONLY the question — no quotes, no explanation."
            )
            res = self.llm.invoke([HumanMessage(content=prompt)])
            text = res.content if isinstance(res.content, str) else str(res.content)
            text = text.strip().strip('"').strip()
            return text if text else question
        except Exception:
            return question

    def _answer_messages(self, question, docs):
        context = "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
        )
        prompt = f"""Answer this question using ONLY the document excerpts below.
Write in simple, clear human language. Cite the page number(s) you used, like (Page 12).
If the answer is not in these excerpts, say exactly: "I don't have enough information to answer that question based on the provided documents."

Question: {question}

Document excerpts:
{context}
"""
        return [SystemMessage(content="You are a helpful assistant that answers questions strictly from the provided PDF document.")] \
            + self.history + [HumanMessage(content=prompt)]

    def _remember(self, question, answer):
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        self.history = self.history[-6:]  # keep last 3 turns only

    # -- step 3: answer ------------------------------------------------------
    def ask(self, question: str):
        standalone = self._condense_question(question)
        self.last_query_used = standalone
        docs = self.retrieve(standalone)
        result = self.llm.invoke(self._answer_messages(question, docs))
        answer = result.content if isinstance(result.content, str) else str(result.content)
        self._remember(question, answer)
        pages_used = sorted({d.metadata.get("page", "?") for d in docs}, key=lambda x: str(x))
        return answer, pages_used, docs

    # -- step 3b: streaming answer -------------------------------------------
    def ask_stream(self, question: str):
        """Same pipeline as ask(), but yields events so the UI can type the answer live:
            {"type": "context", "docs": [...], "pages": [...], "mode": str, "searched_as": str|None}
            {"type": "token", "text": "..."}   (many)
            {"type": "done", "answer": "..."}
        Retrieval (condense + hybrid + rerank) happens first; only the final
        answer generation is streamed, token by token."""
        standalone = self._condense_question(question)
        self.last_query_used = standalone
        docs = self.retrieve(standalone)
        pages_used = sorted({d.metadata.get("page", "?") for d in docs}, key=lambda x: str(x))
        yield {
            "type": "context",
            "docs": docs,
            "pages": pages_used,
            "mode": self.last_mode,
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

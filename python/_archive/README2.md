

## The idea
A PDF can be 100+ pages (an insurance policy, a company policy, a report).
Reading all of it to find one answer is slow. With this project you:

1. **Upload / give the PDF once** — it is split into small chunks and stored in ChromaDB (this is "indexing").
2. **Ask in normal human language** — e.g. *"What is the waiting period for maternity?"*
3. The system searches only the relevant chunks and GPT answers **from your PDF only**,
   and tells you the **page number(s)** the answer came from.

## Files

| File | What it is |
|---|---|
| `pdf_rag_app.py` | **MAIN APP (terminal)** — give it a PDF, ask questions |
| `app_streamlit.py` | **MAIN APP (browser UI)** — upload a PDF in the browser, ask questions |
| `rag_core.py` | Shared engine used by both apps (load PDF → chunk → embed → ChromaDB → answer) |
| `01_ingestion_pipeline.py` | Lesson 1 — load `.txt` docs, chunk, embed, store in ChromaDB |
| `02_retrieval_pipeline.py` | Lesson 2 — retrieve relevant chunks for a query |
| `03_answer_generation.py` | Lesson 3 — generate an answer from retrieved chunks |
| `04_history_aware_generation.py` | Lesson 4 — chat with conversation history |
| `05_recursive_character_text_splitter.py` | Lesson 5 — why recursive splitting beats plain splitting |
| `06_semantic_chunking.py` | Lesson 6 — chunk by meaning (needs an OpenAI key) |
| `07_agentic_chunking.py` | Lesson 7 — let the LLM decide chunk boundaries |
| `08_multi_modal_rag.py` | Lesson 8 — **converted from notebook** — text + tables + images from a PDF (Unstructured + GPT-4o vision; heavy, optional) |
| `09_retrieval_methods.py` | Lesson 9 — similarity / threshold / MMR retrieval |
| `10_multi_query_retrieval.py` | Lesson 10 — generate query variations, search with each |
| `11_reciprocal_rank_fusion.py` | Lesson 11 — fuse multi-query results with RRF |
| `12_hybrid_search.py` | Lesson 12 — **converted from notebook** — vector + BM25 keyword search combined |
| `13_reranker.py` | Lesson 13 — **converted from notebook** — hybrid search + Cohere reranking (needs a Cohere key too) |

Lessons 01–04 and 09–11 use the sample company files in `../docs/` (Google, Microsoft,
Nvidia, SpaceX, Tesla `.txt` files). Run lesson 01 first — it creates `db/chroma_db`,
which the other lessons read.

## Setup (Windows / Mac / Linux)

```bash
cd python
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env     # Windows (Mac/Linux: cp .env.example .env)
notepad .env               # put your real key: OPENAI_API_KEY=sk-...
```

## Run the main app — terminal

```bash
python pdf_rag_app.py --pdf "C:\path\to\your-policy.pdf"
# then type questions in plain English, 'quit' to exit

python pdf_rag_app.py --pdf policy.pdf --question "What is the waiting period for maternity?"
```

First run indexes the PDF (a 100+ page PDF takes a minute or two and uses a little
embedding credit). The index is saved in `python/db/<pdf-name>/`, so the next run is
instant and free. If you replace the PDF, add `--rebuild`.

## Run the main app — browser UI (upload a PDF)

```bash
streamlit run app_streamlit.py
```

Open the local URL it prints, upload your PDF, wait for indexing, and chat.

## Models used (same as the original repo)
- Embeddings: `text-embedding-3-small`
- Answers: `gpt-4o-mini` by default (cheap) — set `OPENAI_MODEL=gpt-4o` in `.env` for the stronger model used in the lessons.

## Verified
- All 13 lesson files + both apps pass a Python syntax check.
- `rag_core` PDF pipeline tested locally: the repo's 15-page sample PDF → 52 chunks with
  correct page numbers, and a synthetic **120-page** policy PDF → 120 chunks, with a clause
  planted on page 57 retrieved with the correct page citation.
- The embedding + GPT answer step needs your real OpenAI key (it calls the OpenAI API).

## Frontend (React, light mode) + API server
The `../frontend/` folder is a React UI in the style of the tutorial videos
(projects sidebar, conversations, Knowledge Base panel). Clicking a source PDF
opens the Processing Pipeline popup — Upload → Partitioning → Chunking →
Summarisation (skipped in fast mode) → Vectorization & Storage → View Chunks
with a Detail Inspector. **No S3, no queue** — all local.

```bash
# terminal 1 (backend API for the UI)
cd python && uvicorn api_server:app --port 8000
# terminal 2 (frontend)
cd frontend && npm install && npm run dev   # http://localhost:5173
```
`api_server.py` reuses `rag_core.py`: upload → partitioning stats (pypdf) →
chunks (saved as JSON in `python/data/` for the chunk viewer) → OpenAI
embeddings into ChromaDB when a key is set → chat with page citations.

## Retrieval upgrade — hybrid search + reranker (integrated)
The chat pipeline in `rag_core.py` no longer uses plain vector search. Per question:
1. **Hybrid search** (lesson 12): Chroma vector search + BM25 keyword search over the
   same PDF's chunks, fused with `EnsembleRetriever` (0.7 vector / 0.3 BM25), top 12 candidates.
   BM25 catches exact clause numbers, amounts and terms that vector search can miss.
2. **Rerank** (lesson 13): candidates are re-ordered to the best 6 — Cohere Rerank if
   `COHERE_API_KEY` is in `.env`, otherwise a GPT-4o-mini listwise rerank (no extra key).
   If reranking fails for any reason, the hybrid order is used — chat never breaks.
3. **Answer** from those 6 chunks, with page citations. The UI shows the retrieval mode
   used under each answer (e.g. "hybrid (vector + BM25) + LLM rerank").
No new packages are needed — `rank-bm25` and `langchain-cohere` are already in requirements.txt.

## Chat history per PDF (conversations)
Every document has its OWN conversations, stored on the backend in
`python/data/conversations/` as JSON — history survives server restarts and page refreshes.
- Many chats per PDF; a chat's title becomes its first question.
- API: `GET/POST /api/documents/{id}/conversations`, `GET/DELETE /api/conversations/{id}`,
  `POST /api/conversations/{id}/chat` (the PdfChat memory is seeded from the saved messages,
  so follow-up questions still work after a restart).
- `DELETE /api/documents/{id}` removes the PDF completely: uploaded file, chunk data,
  ChromaDB store and all of its chats.

## Follow-up aware search + answer sources
- **Query rewriting (condense step):** before retrieving, `PdfChat` rewrites a follow-up
  question into a standalone one using the conversation history
  ("and what about its cost?" -> "What is the premium for the family floater plan?").
  Retrieval + reranking run on the rewritten question; the answer still responds to the
  original wording. The API returns it as `searched_as` and the UI shows it under the answer.
- **Sources:** every answer returns `sources` — the exact chunks used
  ({chunk_id, page, preview}), matching the numbering in the pipeline's View Chunks tab.
  Clicking a source in the UI opens that chunk highlighted. Sources are saved inside the
  conversation history, so they are still there when a chat is reopened later.

## Streaming answers
Chat in the web UI streams: `POST /api/conversations/{id}/chat/stream` returns
Server-Sent Events — first a `meta` event (pages, sources, retrieval mode, searched_as),
then many `token` events, finally `done`. Retrieval still happens first (condense ->
hybrid -> rerank); only the answer generation streams, via `PdfChat.ask_stream`.
When the stream completes, the full answer is saved to the conversation history
exactly like the non-streaming endpoint (which is kept as a fallback).

## Add a source from a URL
`POST /api/upload-url {"url": "..."}` (the "Add website" box in the UI) fetches a public
web page or a PDF link and runs the SAME pipeline as an upload:
- HTML pages: a small built-in extractor (no new dependency) pulls the readable text
  (title kept as the source name), the text is saved as a .txt in uploads/ and segmented
  into ~2500-char sections — for web sources, "Page N" in citations means section N.
- PDF links: downloaded and processed exactly like an uploaded PDF.
- Safety/limits: only http(s), localhost and literal private IPs are refused; pages that
  need JavaScript or a login return a clear error instead of an empty document.
The resulting source supports everything a PDF does: pipeline popup, per-source chats,
streaming, sources, delete.

## Multi-modal: tables, images, scanned pages (no longer text-only)
Deliberately NOT lesson 08's Unstructured stack (Poppler/Tesseract/libmagic are painful
on Windows). Same capability with one pure-pip package (PyMuPDF) + GPT-4o vision:
- **Tables** — PyMuPDF `find_tables` extracts each table as a structured chunk
  ("Plan | Annual Premium" rows), typed `table`, with its page number. Works without a key.
- **Images** — embedded images (>=120px, max 15 per PDF) are described by GPT-4o vision;
  each description becomes a searchable chunk typed `image` ("[Image on page N] ...").
- **Scanned pages** — a page with no text layer is rendered and OCR'd by GPT-4o vision
  (max 30 pages per PDF), and the transcription becomes that page's text before chunking.
  Without a key, scanned pages are flagged in the doc's `multimodal` stats instead of
  failing silently.
Everything lands in the same chunk list, so hybrid search, reranking, sources and the
View Chunks filters (Text / Image / Table) all cover them. The pipeline popup's
Summarisation step now shows what vision processing did. Re-upload a PDF to apply the
new extraction to it. Tunables in .env: OPENAI_VISION_MODEL, OCR_MAX_PAGES, IMAGE_MAX_DESCRIBE.

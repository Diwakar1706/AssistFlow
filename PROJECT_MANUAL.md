# Project Manual — Multi-Modal RAG

**What this project is:** a web app where you upload a PDF (or paste a website URL) and then chat with it in plain English. Answers come only from that document, with page numbers and the exact source chunks shown. This manual explains **every folder and every file** — what it means, what code is inside it, and **the order in which the files were built, first to last**.

---

## Part 1 — The build order (which file was coded 1st, 2nd, 3rd … till finish)

The project was built in layers. Each layer needed the previous one to exist.

| Step | What was built | Why at this position |
|---|---|---|
| **0** | The original tutorial repo (root lesson files `1_….py`–`11_….py`, notebooks `8/12/13`, `docs/` sample files) | The starting point — someone else's learning material, not the app |
| **1** | `python/01_….py` to `python/13_….py` — all 13 lessons converted to plain Python files | First understand every RAG concept as runnable code before building anything |
| **2** | `python/rag_core.py` — the RAG engine | The heart. Everything else (CLI, API, UI) is just a different face on this one file |
| **3** | `python/pdf_rag_app.py` — command-line app | The fastest way to prove the engine works: type a question in the terminal, get an answer |
| **4** | `python/app_streamlit.py` — a quick browser UI | A 1-file web UI to test the engine visually before investing in a real frontend |
| **5** | `python/api_server.py` — FastAPI backend | The real frontend needs an API to talk to; this wraps the engine in HTTP endpoints |
| **6** | `frontend/` — the React app (`api.js` → `App.jsx` → `PipelineModal.jsx` → `styles.css`) | The real UI, built last because it depends on the API's exact shape |
| **7** | Feature upgrades, in this order: hybrid search + reranker (inside `rag_core.py`) → per-PDF chat history (`api_server.py` + `App.jsx`) → follow-up question rewriting + source chunks (`rag_core.py`) → streaming answers (SSE) → website-URL sources (`api_server.py`) → multi-modal tables/images/OCR (`rag_core.py`) | Each upgrade touched the engine first, then the API, then the UI — the same layer order as steps 2 → 5 → 6 |
| **8** | `README.md`, screenshots in `docs/screenshots/`, and this manual | Documentation comes last, when there is something finished to document |

**The one rule of the whole project:** `rag_core.py` knows nothing about HTTP or React, and the frontend knows nothing about PDFs. `api_server.py` is the translator between them.

---

## Part 2 — The runtime flow (what happens when the app runs)

**When you upload a PDF or add a URL:**

```
1. frontend/src/App.jsx        you drop the file / paste the URL
2. frontend/src/api.js         sends it to the backend (POST /api/upload or /api/upload-url)
3. python/api_server.py        saves the file into python/uploads/, creates a document record
4. python/rag_core.py          load_and_split_pdf() reads it page by page:
                                 · normal text → text chunks
                                 · tables (PyMuPDF) → "cell | cell" chunks
                                 · scanned pages → rendered to images → GPT-4o Vision OCR
                                 · embedded images → GPT-4o Vision writes descriptions
                               then RecursiveCharacterTextSplitter cuts 1000-char chunks (200 overlap)
5. python/rag_core.py          get_vector_store() embeds every chunk (text-embedding-3-small)
                               and stores them in python/db/<pdf-id>/ (ChromaDB, one store per PDF)
6. frontend PipelineModal      shows the 5 steps and lets you inspect every chunk
```

**When you ask a question:**

```
1. App.jsx → api.js            POST /api/conversations/{id}/chat/stream
2. api_server.py               loads that conversation's saved history from python/data/conversations/
3. rag_core.py PdfChat         _condense_question() — if it's a follow-up ("and its cost?"),
                               rewrites it into a full question using the history
4. rag_core.py                 _retrieve_candidates() — hybrid search: vector search + BM25
                               keyword search (EnsembleRetriever, 70/30) → top 12 chunks
5. rag_core.py                 _rerank() — Cohere (if key set) or GPT-4o-mini reorders them → best 6
6. rag_core.py                 ask_stream() — GPT answers using ONLY those 6 chunks,
                               streamed token-by-token (SSE) back to the browser
7. api_server.py               saves the finished Q&A (with its source chunks) into the conversation file
8. App.jsx                     shows the answer, "Based on page(s)…", and clickable source chips
```

---

## Part 3 — Folder by folder

### 📁 Root folder — `multi-model-rag/`

The project home. It contains only project-level files and the two real sub-projects (`python/` and `frontend/`) — nothing else. **Every file exists in exactly one place** (fixed 2026-11-15: the original tutorial lesson files that used to sit here as a second copy were removed — they were byte-identical duplicates of the `01`–`13` lessons in `python/`, and the original notebooks 8/12/13 exist there as converted `.py` files).

| File | Meaning |
|---|---|
| `README.md` | The public face of the project — what it is, screenshots, how to start it. This is what GitHub shows |
| `PROJECT_MANUAL.md` | This file — the complete internal guide to every folder and file |
| `.gitignore` | Tells Git what never to upload: `.env` (contains the API key), `node_modules`, runtime data |
| `docs/` | Sample documents + screenshots + test questions (see below) |
| `python/` | The backend and the 13 lessons (see below) |
| `frontend/` | The React UI (see the frontend section) |

### 📁 `docs/` — sample documents and screenshots

| File | Meaning |
|---|---|
| `attention-is-all-you-need.pdf` | The famous 15-page Transformer paper — the main test document for the whole project |
| `Google.txt`, `Microsoft.txt`, `Nvidia.txt`, `SpaceX.txt`, `Tesla.txt` | Small company-profile text files. The early lessons (01–04) load these to learn ingestion and retrieval |
| `synthetic_questions.txt` | A ready-made list of test questions about the attention paper — used to test the app by hand (moved here from the root in the 2026-11-15 structure cleanup) |
| `screenshots/chat-with-web-source.png` | Screenshot of the running app chatting with the Wikipedia URL source (used in the README) |
| `screenshots/per-pdf-chat-history.png` | Screenshot showing separate saved conversations per PDF (used in the README) |

### 📁 `python/` — the backend and the lessons

Everything Python lives here: the 13 lessons, the RAG engine, and three different "faces" on the engine (CLI, Streamlit, FastAPI). Three more folders appear here **only after you run the app** — they are created at runtime:

| Runtime folder | Meaning |
|---|---|
| `python/uploads/` | Every PDF you upload (and every web page, saved as a `.txt`) is stored here |
| `python/db/<pdf-id>/` | The ChromaDB vector store for one document — its chunks turned into embeddings. One folder per PDF, named by a hash of the file, so re-uploading the same PDF reuses its store |
| `python/data/conversations/` | One small JSON file per chat — the saved questions, answers and sources, so history survives restarts |

| File | Meaning |
|---|---|
| `.env.example` | Template for your settings. Copy it to `.env` and put your real `OPENAI_API_KEY` inside. `.env` itself is never shared or uploaded |
| `requirements.txt` | The exact Python packages the backend needs (langchain, chromadb, fastapi, uvicorn, pypdf, PyMuPDF, rank-bm25, cohere…). This is the **only** requirements file to install — it lives here because you run everything from this folder |
| `requirements-original-freeze.txt` | The original author's full pip-freeze from his Mac/Linux machine. Reference only — do **not** install it on Windows (it contains Linux-only packages like `uvloop` that fail) |
| `README.md` | Short guide for just the python folder: the lessons list and how to run the CLI app |

#### The 13 lessons (`01`–`13`) — the learning layer

Each lesson is a small, standalone, runnable file that teaches **one** RAG concept. The app itself is these concepts combined.

| File | Concept it teaches |
|---|---|
| `01_ingestion_pipeline.py` | **Ingestion** — load text files, split them into chunks, embed them, store them in ChromaDB. The first step of every RAG system |
| `02_retrieval_pipeline.py` | **Retrieval** — load the stored vectors and find the chunks most similar to a question |
| `03_answer_generation.py` | **Generation** — give the retrieved chunks + question to the LLM and get a grounded answer. Lessons 1+2+3 together = a basic RAG |
| `04_history_aware_generation.py` | **Chat memory** — the retriever uses the conversation history so follow-up questions make sense |
| `05_recursive_character_text_splitter.py` | **Chunking strategy** — how RecursiveCharacterTextSplitter cuts text (the splitter the app uses: 1000/200) |
| `06_semantic_chunking.py` | **Semantic chunking** — cutting text where the *meaning* changes, instead of every N characters |
| `07_agentic_chunking.py` | **Agentic chunking** — an LLM reads the text and decides the chunk boundaries itself (propositions) |
| `08_multi_modal_rag.py` | **Multi-modal RAG** — extracting text + tables + images from a PDF (the tutorial's Unstructured approach; the app uses a lighter PyMuPDF + GPT-4o Vision version of the same idea) |
| `09_retrieval_methods.py` | **Retrieval methods compared** — similarity search, MMR (diversity), and multi-query side by side |
| `10_multi_query_retrieval.py` | **Multi-query** — the LLM rewrites one question into several, searches with all of them, and merges the results |
| `11_reciprocal_rank_fusion.py` | **RRF** — the maths for merging ranked lists from different searches into one ranking |
| `12_hybrid_search.py` | **Hybrid search** — vector (meaning) search + BM25 (keyword) search combined with EnsembleRetriever. *This lesson is inside the app* |
| `13_reranker.py` | **Reranking** — after hybrid search fetches candidates, a reranker re-reads them and reorders by true relevance. *This lesson is also inside the app* |

#### `rag_core.py` — ⭐ the engine (443 lines)

The most important file in the project. It contains the complete RAG pipeline with **no** web code — the CLI, Streamlit app and FastAPI server all call into it. If you understand this file, you understand the project.

Settings at the top (read from `.env`): the chat model (`gpt-4o-mini`), vision model (`gpt-4o`), embedding model (`text-embedding-3-small`), and the multi-modal limits (`OCR_MAX_PAGES=30`, `IMAGE_MAX_DESCRIBE=15`).

| Function / class | What it does |
|---|---|
| `_pdf_id(pdf_path)` | Gives each PDF a short ID (a hash of the file path) — used to name its vector-store folder |
| `_load_text_segments(txt_path)` | Reads a saved web page (`.txt` from a URL source) and cuts it into ~2500-character sections; each section counts as one "page" for citations |
| `_vision_message(prompt, png_bytes)` | Builds one GPT-4o Vision request: a text prompt + one image, in the format the OpenAI API expects |
| `_ocr_page_text(fitz_page)` | Renders one scanned PDF page to a 200-dpi image and asks GPT-4o Vision to read all the text on it (OCR) |
| `_extract_table_docs(fitz_doc, source_name)` | Finds every table in the PDF with PyMuPDF and turns each into a chunk formatted as `cell \| cell \| cell`, tagged with its page number |
| `describe_pdf_images(pdf_path, max_images)` | Pulls embedded images (≥120px, up to the cap) out of the PDF and asks GPT-4o Vision to describe each one; every description becomes a searchable chunk |
| `load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200)` | **The loader.** Reads a PDF page by page (or a web `.txt` by sections). Normal pages → text; nearly-empty pages are treated as scanned → OCR'd (or flagged `scanned` if no API key). Adds table chunks. Then splits everything into 1000-char chunks with 200-char overlap, each keeping its page number |
| `get_vector_store(pdf_path, force_rebuild, extra_chunks)` | **The store.** Embeds all chunks with `text-embedding-3-small` and saves them in ChromaDB under `db/<pdf-id>/`. If the store already exists it is reused (fast re-open); `extra_chunks` (image descriptions) are added on a rebuild |
| `class PdfChat` | **The chat brain.** One instance = one document you can talk to. Holds the vector store, the document list (for BM25), the chat model, and the conversation memory |
| `PdfChat._retrieve_candidates(question)` | Hybrid search: EnsembleRetriever combining vector search (weight 0.7) and BM25 keyword search (weight 0.3) → top **12** candidate chunks |
| `PdfChat._rerank(question, candidates)` | Reorders the 12 candidates by true relevance → best **6**. Uses Cohere Rerank if `COHERE_API_KEY` is set, otherwise GPT-4o-mini listwise ranking; if ranking fails for any reason it silently keeps the hybrid order |
| `PdfChat.retrieve(question)` | Public wrapper: candidates → rerank → final chunks |
| `PdfChat._condense_question(question)` | **Follow-up rewriting.** If there is chat history, the LLM rewrites a follow-up like *"and what about its cost?"* into a standalone question (*"What is the premium for the family floater plan?"*) before searching. Remembers the rewritten form so the UI can show "Searched as: …" |
| `PdfChat._answer_messages(question, docs)` | Builds the exact prompt sent to the LLM: a system instruction (answer **only** from the excerpts, cite page numbers, say so if the answer is not in the document) + the chunks + memory + question |
| `PdfChat._remember(question, answer)` | Appends the finished exchange to the conversation memory |
| `PdfChat.ask(question)` | The complete non-streaming path: condense → retrieve → answer → remember. Returns the answer, the pages used, the source chunks, the retrieval mode, and the rewritten question if any |
| `PdfChat.ask_stream(question)` | The same path, but yields events: first the context (pages/sources/mode), then the answer **token by token**, then a done event — this is what makes answers type themselves out live in the UI |

#### `pdf_rag_app.py` — the command-line app (62 lines)

The simplest face on the engine. Run `python pdf_rag_app.py --pdf <file>` and it loads the PDF, builds/loads its vector store, and starts a terminal chat loop (`main()`): you type a question, it prints the answer with its pages; type `exit` to stop. Built at step 3 to prove the engine works with zero web code.

#### `app_streamlit.py` — the quick web UI (53 lines)

A one-file Streamlit interface over the same engine: upload a PDF in the sidebar, chat in the main area. Built at step 4 as a fast visual test. The React app later replaced it as the real UI, but it still works and is handy for demos without starting two servers.

#### `api_server.py` — ⭐ the FastAPI backend (624 lines)

The translator between the React frontend and `rag_core.py`. It owns everything about **documents, uploads, URLs, and saved conversations**, and exposes **13 REST endpoints**. Started with `uvicorn api_server:app --port 8000`.

Helper functions (internal plumbing):

| Function | What it does |
|---|---|
| `_partition_stats(pdf_path)` | Counts pages, text characters, and tables in a PDF — feeds the pipeline popup's Partitioning step |
| `_doc_file / _save_doc / _load_doc` | Reads and writes a document's record (name, size, kind, chunk count, pipeline steps, multi-modal stats) as JSON in `data/` |
| `_process_source(doc_id, file_path, …)` | **The upload pipeline.** Calls `rag_core` to load, split, OCR/describe, embed and store a new source; records the real numbers for every pipeline step (pages, chunks, tables, images, OCR pages) and caches a `PdfChat` for it |
| `_text_stats(text, num_segments)` | Word/character statistics for web-page sources |
| `_assert_public_url(url)` | URL safety check: only `http(s)`, refuses localhost, `.local` names and private IP addresses |
| `_ReadableText` + `_html_to_text(html)` | A small readable-text extractor built on Python's standard library: strips scripts, styles, navigation and footers from a web page and keeps the article text + title |
| `_sources_for(doc, docs)` | Turns the chunks an answer used into the source list the UI shows (`chunk_id`, page, preview) |
| `_conv_file / _save_conv / _load_conv / _conv_summary / _list_convs` | Reads, writes and lists the saved conversation JSON files; a summary is the lightweight version (title, message count, last-active time) shown in the UI list |
| `_get_conv_chat(conv)` | Rebuilds a `PdfChat` for a conversation and **seeds its memory from the saved messages**, so follow-ups still work after a server restart |

The endpoints (what the frontend is allowed to ask for):

| Endpoint | Meaning |
|---|---|
| `GET /api/health` | "Are you up, and is an OpenAI key set?" — drives the green dots in the sidebar |
| `GET /` and `GET /health` | Friendly aliases so opening the backend address in a browser doesn't show a scary 404 |
| `GET /api/documents` | List all sources in the Knowledge Base |
| `GET /api/documents/{id}` | One document's full record, including its chunks (the pipeline popup reads this) |
| `POST /api/upload` | Upload a PDF → runs the whole processing pipeline |
| `POST /api/upload-url` | Add a website URL or PDF link → fetched, extracted, processed like an upload |
| `POST /api/chat` | Simple one-shot question against a document (kept for the CLI-style flow) |
| `GET /api/documents/{id}/conversations` | List a document's saved chats |
| `POST /api/documents/{id}/conversations` | Start a new chat for a document |
| `GET /api/conversations/{id}` | Load one chat with all its messages |
| `DELETE /api/conversations/{id}` | Delete one chat |
| `POST /api/conversations/{id}/chat` | Ask a question in a chat (whole answer at once) |
| `POST /api/conversations/{id}/chat/stream` | Ask a question in a chat with the answer **streamed** over SSE — the one the UI actually uses |
| `DELETE /api/documents/{id}` | Delete a document completely: file, chunks, vector store, and all its chats |

### 📁 `frontend/` — the React web app

A standard Vite + React project. It contains **no RAG logic at all** — it only displays what the API returns and sends what you type. Start it with `npm run dev` (port 5173).

| File | Meaning |
|---|---|
| `package.json` | The project's ID card for Node: name (`multi-model-rag-frontend`), the scripts (`npm run dev` = start, `npm run build` = production build), and the dependencies (react, react-dom, vite) |
| `package-lock.json` | The exact locked version of every installed package, so installs are identical on every machine. Auto-generated — never edit by hand |
| `vite.config.js` | Vite's settings. The important line: `/api` requests are **proxied** to the backend on port 8000, so the browser only ever talks to one address |
| `index.html` | The single HTML page of the app — an empty shell with a `<div id="root">` that React fills in |
| `README.md` | The default Vite template readme (kept from project creation) |

#### 📁 `frontend/src/` — the app source

| File | Meaning |
|---|---|
| `main.jsx` (8 lines) | The entry point. Takes `App` and mounts it into `index.html`'s root div. The first JS that runs |
| `api.js` (111 lines) | **The frontend's phone book** — one function per backend endpoint, so the rest of the UI never writes a URL by hand. `fetchHealth`, `fetchDocuments`, `fetchDocument`, `uploadPdf`, `addUrl`, `fetchConversations`, `createConversation`, `fetchConversation`, `deleteConversation`, `deleteDocument`, `askQuestion`, `askInConversation`, and `askInConversationStream` — the last one reads the SSE stream piece by piece and calls a callback for every event, which is how the answer types itself live |
| `App.jsx` | **The whole main screen.** Fixed layout (header and footer never scroll — only the chat messages scroll): left **History** sidebar (every source by name — PDF or website URL — with its chats listed under it and a 🗑 delete on each chat, a ＋ New chat button, and the backend + key status dots), centre panel (fixed header showing **only the selected source's name**, the scrolling chat messages with their answer sources in uniform-size chunk boxes, and the fixed question input as the footer), right Knowledge Base (upload dropzone, URL box, sources list). *(Layout updated 2026-11-15 at the user's request: history moved from the top of the centre panel into the left sidebar, replacing the old "GenAI Team" project card.)* It holds all the screen's state — which document is selected, which chat is open, the messages — and the small helpers `fmtSize` (bytes → "3.3 MB"), `timeAgo` ("17m ago") and `statusOf` (is this source Ready?). It also auto-creates a conversation when you type with none open, and opens the pipeline popup on the exact chunk when you click a source chip under an answer |
| `PipelineModal.jsx` (248 lines) | **The processing-pipeline popup** that opens when you click a source's 👁. Shows the 5 steps — Upload → Partitioning → Chunking → Summarisation → Vectorization & Storage — each with the document's *real* numbers (pages, tables found, chunks, images described, OCR pages), plus **View Chunks**: every chunk in a searchable, filterable list with a detail inspector, and the chunk an answer used highlighted in amber |
| `styles.css` (172 lines) | All the visual styling for both files above: the light theme, the three columns, chat bubbles, buttons, the popup, the blinking cursor while an answer streams. Pure appearance — no logic |

---

## Part 4 — How the pieces fit together (one picture)

```
        ┌──────────────────────── frontend/ (React, :5173) ───────────────────────┐
        │  main.jsx → App.jsx (screen) ──uses──► api.js (calls) ──uses──► PipelineModal.jsx (popup)  │
        └──────────────────────────────────┬───────────────────────────────────────┘
                                           │  HTTP / SSE   (vite proxies /api → :8000)
        ┌──────────────────────── python/ (FastAPI, :8000) ───────────────────────┐
        │  api_server.py  (documents · uploads · URLs · saved conversations)      │
        └──────────────────────────────────┬───────────────────────────────────────┘
                                           │  plain Python calls
        ┌──────────────────────── python/rag_core.py (the engine) ────────────────┐
        │  load & split → embed → ChromaDB │ PdfChat: hybrid search → rerank → GPT │
        └──────────────────────────────────┬───────────────────────────────────────┘
                                           │
              OpenAI API (embeddings · answers · vision/OCR)   +   disk: uploads/ · db/ · data/
```

The lessons (`01`–`13`) sit beside this picture: they are where each idea inside `rag_core.py` was first learned — chunking (05), history (04), hybrid search (12), reranking (13), multi-modal (08).

## Part 5 — Small glossary (interview-ready)

- **Chunk** — a small piece of a document (~1000 characters). We search chunks, not whole documents, so the LLM reads only what matters.
- **Embedding** — a list of numbers representing a chunk's *meaning*. Similar meanings get similar numbers, which is what makes semantic search possible.
- **Vector store (ChromaDB)** — the database that keeps embeddings and finds the nearest ones to a question's embedding.
- **BM25** — classic keyword search: matches exact words. Weak on meaning, strong on exact terms like "Clause 42B".
- **Hybrid search** — vector + BM25 together, so both meaning-matches and exact-word matches are found.
- **Reranker** — a second, slower, more careful judge that re-reads the top candidates and reorders them by true relevance.
- **OCR** — turning a picture of text (a scanned page) back into real text — here, done by GPT-4o Vision.
- **SSE (Server-Sent Events)** — the browser keeps one request open and the server pushes the answer token by token; that's the live typing effect.

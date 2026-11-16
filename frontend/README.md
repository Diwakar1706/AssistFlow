# Frontend — RAGBase (React + Vite, light mode)

Layout mirrors the tutorial UI: left = projects, centre = conversations + chat,
right = Knowledge Base (upload + sources). Clicking a source PDF opens the
**Processing Pipeline** popup: Upload → Partitioning → Chunking → Summarisation
(skipped in local fast mode) → Vectorization & Storage → View Chunks with a
Detail Inspector. No S3 and no queue — everything runs locally.

## Run it

Terminal 1 — backend (from the project root):
```bash
cd python
pip install -r requirements.txt
# put OPENAI_API_KEY in python/.env  (copy .env.example to .env)
uvicorn api_server:app --port 8000
```

Terminal 2 — frontend:
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

- Upload a PDF in the Knowledge Base panel → the pipeline popup opens with the real numbers.
- "Chat with this PDF" → ask questions in plain English; answers cite page numbers.
- Without an OpenAI key, everything works except the Vectorization step and chat
  (the UI tells you exactly that — chunks are still viewable).

## Per-PDF chat history
Select a PDF in the Knowledge Base and the centre panel shows THAT PDF's conversations.
Use "+ New conversation" for a second chat on the same PDF, click a chat to reopen its full
history, 🗑 on a chat to delete just that chat, and 🗑 on a source PDF to remove the PDF
plus all its chats. History is stored by the backend, so it survives refreshes and restarts.

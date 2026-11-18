"""
AssistFlow — FastAPI application entry point.

Start the server:
    cd python
    uvicorn server:app --port 8000

The React frontend (frontend/) proxies all /api calls to this server.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.documents     import router as documents_router
from api.routes.conversations  import router as conversations_router
from api.routes.chat           import router as chat_router

app = FastAPI(title="AssistFlow — RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers under /api
app.include_router(documents_router,    prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(chat_router,         prefix="/api")


# ── Health & root ──────────────────────────────────────────────────────

@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "ok":              True,
        "openai_key_set":  bool(os.getenv("OPENAI_API_KEY")),
        "cohere_key_set":  bool(os.getenv("COHERE_API_KEY")),
    }


@app.get("/")
def root():
    return {
        "service":         "AssistFlow — RAG API",
        "status":          "running",
        "openai_key_set":  bool(os.getenv("OPENAI_API_KEY")),
        "frontend":        "Open the React UI at http://localhost:5173",
        "docs":            "http://localhost:8000/docs",
    }

"""
Centralized configuration — every setting, path, and constant in one place.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_ROOT   = os.path.join(BASE_DIR, "db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR  = os.path.join(BASE_DIR, "data")
CONV_DIR  = os.path.join(DATA_DIR, "conversations")

# Ensure runtime directories exist on import
for _dir in (DB_ROOT, UPLOAD_DIR, DATA_DIR, CONV_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL",  "text-embedding-3-small")
CHAT_MODEL   = os.getenv("OPENAI_MODEL",         "gpt-4o-mini")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL",  "gpt-4o")

# ── Processing limits (cost caps) ─────────────────────────────────────
OCR_MAX_PAGES     = int(os.getenv("OCR_MAX_PAGES",      "30"))
IMAGE_MAX_DESCRIBE = int(os.getenv("IMAGE_MAX_DESCRIBE", "15"))
MIN_IMAGE_PX      = 120   # ignore tiny icons / logos

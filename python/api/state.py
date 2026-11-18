"""
Shared in-memory state for chat session caches.
Kept in one place so all route modules can import it without circular deps.
"""

# doc_id -> PdfChat  (legacy /api/chat endpoint)
_chats: dict = {}

# conversation_id -> PdfChat  (per-conversation endpoint)
_conv_chats: dict = {}

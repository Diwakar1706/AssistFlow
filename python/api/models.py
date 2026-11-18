"""
Pydantic request / response schemas used across the API routes.
"""
from pydantic import BaseModel


class UrlIn(BaseModel):
    url: str


class ChatIn(BaseModel):
    document_id: str
    question:    str


class ConvChatIn(BaseModel):
    question: str

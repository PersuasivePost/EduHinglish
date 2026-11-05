"""Request/response models shared by the backend API (what the frontend sees)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Subject = Literal["biology", "chemistry", "physics", "mathematics", "social_science"]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000,
                       examples=["Ohm ka law kya hai aur iska formula kya hai?"])
    subject: Optional[Subject] = None
    class_num: Optional[Literal["9", "10"]] = None
    language: Literal["hinglish", "english"] = "hinglish"


class Source(BaseModel):
    chapter: str
    subject: str = ""
    class_num: str = Field("", validation_alias="class")
    score: float
    chunk_id: str = ""
    text_preview: str = ""


class AskResponse(BaseModel):
    query: str
    answer: str
    language: str
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    sources: list[Source]
    timings: dict[str, float]


class HealthResponse(BaseModel):
    backend: str = "ok"
    model_server_url: str
    model_server: dict


# ── Chat API (/api/chat, /api/chats) ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=1000, examples=["Ohm ka law kya hai?"])
    chat_id: Optional[str] = Field(None, description="Omit to start a new chat")
    subject: Optional[Subject] = None
    class_num: Optional[Literal["9", "10"]] = None
    language: Literal["hinglish", "english"] = "hinglish"


class MessageOut(BaseModel):
    id: str
    chat_id: str
    role: Literal["user", "assistant"]
    content: str
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    sources: list[dict] = []
    timings: dict[str, float] = {}
    feedback: int = 0
    created_at: float


class ChatOut(BaseModel):
    id: str
    title: str
    subject: Optional[str] = None
    class_num: Optional[str] = None
    language: str = "hinglish"
    created_at: float
    updated_at: float


class ChatDetail(ChatOut):
    messages: list[MessageOut]


class ChatResponse(BaseModel):
    chat: ChatOut
    user_message: MessageOut
    assistant_message: MessageOut


class ChatUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=120)
    subject: Optional[Subject] = None
    class_num: Optional[Literal["9", "10"]] = None
    language: Optional[Literal["hinglish", "english"]] = None


class FeedbackRequest(BaseModel):
    value: Literal[-1, 0, 1] = Field(..., description="1 = 👍, -1 = 👎, 0 = clear")

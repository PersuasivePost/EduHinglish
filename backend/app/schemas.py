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

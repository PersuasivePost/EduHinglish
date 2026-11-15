"""
EduHinglish backend API — what the frontend (and later the Flutter app) calls.

It holds no model. Every question is forwarded to the model server (notebook 07
on Colab, or src/model_server.py on a GPU machine) at MODEL_SERVER_URL.
Chats, messages and 👍/👎 feedback are stored in SQLite (DB_PATH).
The chat page in backend/static/ is served at /.

Run (from the repo root):
    pip install -r backend/requirements.txt
    cp backend/.env.example backend/.env      # then fill in MODEL_SERVER_URL + MODEL_API_KEY
    uvicorn backend.app.main:app --reload --port 8000
Chat page: http://localhost:8000      API docs: http://localhost:8000/docs
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db
from .model_client import ModelClient, ModelServerError
from .schemas import (AskRequest, AskResponse, ChatDetail, ChatOut, ChatRequest, ChatResponse, ChatUpdate,
                      FeedbackRequest, HealthResponse, MessageOut)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    app.state.model = ModelClient()
    yield
    await app.state.model.close()


app = FastAPI(title="EduHinglish Backend", version="1.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


def _raise(e: ModelServerError):
    raise HTTPException(status_code=e.status_code, detail=e.detail)


def _chat_or_404(chat_id: str) -> dict:
    chat = db.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


def _title(message: str) -> str:
    message = " ".join(message.split())
    return message if len(message) <= 60 else message[:57].rstrip() + "..."


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    try:
        model = await app.state.model.health()
    except ModelServerError as e:
        model = {"status": "offline", "detail": e.detail}
    return HealthResponse(model_server_url=config.MODEL_SERVER_URL or "(not set)", model_server=model)


# ── Chat ─────────────────────────────────────────────────────────────────────

def _chat_history(req: ChatRequest) -> list[dict]:
    """Earlier messages of this chat, so follow-up questions ("aur iska formula?") are understood."""
    if not req.chat_id:
        return []
    _chat_or_404(req.chat_id)
    return [{"role": m["role"], "content": m["content"]}
            for m in db.list_messages(req.chat_id)[-config.HISTORY_MESSAGES:]]


def _model_payload(req: ChatRequest, history: list[dict]) -> dict:
    return {"query": req.message, "subject": req.subject, "class_num": req.class_num,
            "language": req.language, "history": history}


def _save_turn(req: ChatRequest, res: dict) -> ChatResponse:
    """Store the question and the model's answer (only called once the model has answered)."""
    settings = {"subject": req.subject, "class_num": req.class_num, "language": req.language}
    if req.chat_id:
        chat = db.update_chat(req.chat_id, **settings)
    else:
        chat = db.create_chat(_title(req.message), **settings)
    user_msg = db.add_message(chat["id"], "user", req.message)
    bot_msg = db.add_message(chat["id"], "assistant", res.get("answer", ""),
                             intent=res.get("intent"), intent_confidence=res.get("intent_confidence"),
                             sources=res.get("sources", []), timings=res.get("timings", {}))
    return ChatResponse(chat=db.get_chat(chat["id"]), user_message=user_msg, assistant_message=bot_msg,
                        standalone_query=res.get("standalone_query"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message. Creates a new chat when chat_id is omitted.

    With a chat_id, the last few messages are sent along so follow-up questions
    are understood. The model is called first; nothing is saved if it fails,
    so the client can simply retry.
    """
    history = _chat_history(req)
    try:
        res = await app.state.model.ask(_model_payload(req, history))
    except ModelServerError as e:
        _raise(e)
    return _save_turn(req, res)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Same as /api/chat but streamed as Server-Sent Events, so the answer appears as it is written.

    Events (one `data: {json}` line each): status {stage}, meta {standalone_query, intent, sources},
    token {text}, then done {chat, user_message, assistant_message, standalone_query} or
    error {status, detail}. Nothing is saved unless "done" is sent.
    """
    history = _chat_history(req)   # 404 for an unknown chat happens before the stream starts

    async def events():
        try:
            async for ev in app.state.model.ask_stream(_model_payload(req, history)):
                if ev.get("type") == "done":
                    saved = _save_turn(req, ev)
                    ev = {"type": "done", **saved.model_dump(mode="json")}
                elif ev.get("type") == "error":
                    ev = {"type": "error", "status": 502, "detail": f"Model server: {ev.get('detail')}"}
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except ModelServerError as e:
            yield f"data: {json.dumps({'type': 'error', 'status': e.status_code, 'detail': e.detail})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/chats", response_model=list[ChatOut])
def list_chats():
    return db.list_chats()


@app.get("/api/chats/{chat_id}", response_model=ChatDetail)
def get_chat(chat_id: str):
    return {**_chat_or_404(chat_id), "messages": db.list_messages(chat_id)}


@app.patch("/api/chats/{chat_id}", response_model=ChatOut)
def update_chat(chat_id: str, req: ChatUpdate):
    _chat_or_404(chat_id)
    return db.update_chat(chat_id, **req.model_dump(exclude_unset=True))


@app.delete("/api/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: str):
    if not db.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")


@app.post("/api/messages/{message_id}/feedback", response_model=MessageOut)
def feedback(message_id: str, req: FeedbackRequest):
    msg = db.set_feedback(message_id, req.value)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    if msg["role"] != "assistant":
        raise HTTPException(status_code=400, detail="Feedback is only for assistant messages")
    return msg


# ── Single-shot endpoints (no history) ───────────────────────────────────────

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        res = await app.state.model.ask(req.model_dump())
    except ModelServerError as e:
        _raise(e)
    return AskResponse(**res)


class PreprocessRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=1000)


@app.post("/api/preprocess")
async def preprocess(req: PreprocessRequest):
    """M1 trace (script detection, normalisation, LID, ...) for the 'how it works' view."""
    try:
        return await app.state.model.preprocess(req.sentence)
    except ModelServerError as e:
        _raise(e)


# ── Chat page (mounted last so /api/* and /docs win) ─────────────────────────

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

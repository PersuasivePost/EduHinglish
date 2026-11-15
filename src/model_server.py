"""
EduHinglish — Model Server (runs next to the GPU)
==================================================
Small HTTP API around EduHinglishPipeline (M1 → M2 → M3). It runs on Colab (see
training/colab_notebooks/07_colab_model_server.ipynb) or on any GPU machine.
The project backend (backend/) calls it. The frontend never calls it directly.

Endpoints:
    GET  /health      → status, adapter, knowledge-base size (no key needed)
    POST /ask         → {"query", "subject"?, "class_num"?, "language"?, "top_k"?, "max_new_tokens"?,
                         "history"? = [{"role", "content"}, ...] earlier turns, for follow-up questions}
    POST /ask/stream  → same body as /ask, streams Server-Sent Events (status, meta, token, done)
    POST /preprocess  → {"sentence"}  M1 trace only

Auth: if MODEL_API_KEY is set, /ask and /preprocess need the header  X-API-Key: <key>.

Run on a GPU machine:
    MODEL_API_KEY=secret python src/model_server.py --port 8000
In a notebook (pipeline already loaded):
    import model_server; model_server.start_in_thread(pipe, port=8000)
"""

import json
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

Subject = Literal["biology", "chemistry", "physics", "mathematics", "social_science"]

state: dict = {"pipeline": None, "started_at": None}
gpu_lock = threading.Lock()   # one GPU, one model → one generation at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    if state["pipeline"] is None:   # started from the CLI: load here
        from pipeline import EduHinglishPipeline
        state["pipeline"] = EduHinglishPipeline(mode=os.environ.get("EDUHINGLISH_MODE", "full"))
    state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    yield


app = FastAPI(title="EduHinglish Model Server", version="1.0", lifespan=lifespan)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("MODEL_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key")


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    subject: Optional[Subject] = None
    class_num: Optional[Literal["9", "10"]] = None
    language: Literal["hinglish", "english"] = "hinglish"
    top_k: int = Field(3, ge=1, le=8)
    max_new_tokens: int = Field(256, ge=16, le=768)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12,
                                          description="Earlier turns of this chat, oldest first")


class PreprocessRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health():
    p = state["pipeline"]
    if p is None:
        return {"status": "loading"}
    g = p.generator
    return {
        "status": "ok",
        "mode": p.mode,
        "started_at": state["started_at"],
        "adapter": g.adapter_source if g else None,
        "lora_loaded": bool(g and g.lora_loaded),
        "device": g.device if g else None,
        "intent_model": p.intent_clf is not None,
        "kb_chunks": p.retriever.get_collection_stats()["total_documents"] if p.retriever else 0,
        "busy": gpu_lock.locked(),
    }


def _ready_pipeline():
    p = state["pipeline"]
    if p is None or p.mode != "full":
        raise HTTPException(status_code=503, detail="Model not loaded (preprocess-only mode)")
    return p


def _run_answer(p, req: AskRequest, on_event=None) -> dict:
    with gpu_lock:
        r = p.answer_query(req.query, top_k=req.top_k, subject=req.subject, class_num=req.class_num,
                           language=req.language, max_new_tokens=req.max_new_tokens,
                           history=[m.model_dump() for m in req.history], on_event=on_event)
    intent = r.get("intent") or {}
    return {
        "query": req.query,
        "standalone_query": r.get("standalone_query", req.query),
        "answer": r["generation"]["hinglish_answer"],
        "language": req.language,
        "intent": intent.get("intent"),
        "intent_confidence": intent.get("confidence"),
        "intent_used_in_prompt": r["generation"]["prompt_intent"],
        "sources": r["retrieval"],
        "timings": {t["step"]: t["time_sec"] for t in r["pipeline_trace"]},
    }


@app.post("/ask", dependencies=[Depends(check_key)])
def ask(req: AskRequest):
    return _run_answer(_ready_pipeline(), req)


@app.post("/ask/stream", dependencies=[Depends(check_key)])
def ask_stream(req: AskRequest):
    """Same as /ask but streams Server-Sent Events: status, meta, token..., then done (or error).

    Each event is one line `data: {json}` followed by a blank line; "done" carries the /ask payload.
    """
    p = _ready_pipeline()
    events: queue.Queue = queue.Queue()

    def worker():
        try:
            if gpu_lock.locked():
                events.put({"type": "status", "stage": "queued"})
            events.put({"type": "done", **_run_answer(p, req, on_event=events.put)})
        except Exception as e:   # includes ValueError for an unknown subject
            events.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def sse():
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/preprocess", dependencies=[Depends(check_key)])
def preprocess(req: PreprocessRequest):
    p = state["pipeline"]
    if p is None:
        raise HTTPException(status_code=503, detail="Still loading")
    return json.loads(json.dumps(p.process(req.sentence), default=str, ensure_ascii=False))


def start_in_thread(pipeline, host: str = "0.0.0.0", port: int = 8000):
    """Serve an already-loaded pipeline from a background thread (for notebooks)."""
    import uvicorn

    state["pipeline"] = pipeline
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.2)
    print(f"[OK] Model server on http://{host}:{port}")
    return server


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

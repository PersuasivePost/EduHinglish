"""
EduHinglish — Model Server (runs next to the GPU)
==================================================
Small HTTP API around EduHinglishPipeline (M1 → M2 → M3). It runs on Colab (see
training/colab_notebooks/07_colab_model_server.ipynb) or on any GPU machine.
The project backend (backend/) calls it. The frontend never calls it directly.

Endpoints:
    GET  /health      → status, adapter, knowledge-base size (no key needed)
    POST /ask         → {"query", "subject"?, "class_num"?, "language"?, "top_k"?, "max_new_tokens"?}
    POST /preprocess  → {"sentence"}  M1 trace only

Auth: if MODEL_API_KEY is set, /ask and /preprocess need the header  X-API-Key: <key>.

Run on a GPU machine:
    MODEL_API_KEY=secret python src/model_server.py --port 8000
In a notebook (pipeline already loaded):
    import model_server; model_server.start_in_thread(pipe, port=8000)
"""

import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, Header, HTTPException
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


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    subject: Optional[Subject] = None
    class_num: Optional[Literal["9", "10"]] = None
    language: Literal["hinglish", "english"] = "hinglish"
    top_k: int = Field(3, ge=1, le=8)
    max_new_tokens: int = Field(256, ge=16, le=768)


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


@app.post("/ask", dependencies=[Depends(check_key)])
def ask(req: AskRequest):
    p = state["pipeline"]
    if p is None or p.mode != "full":
        raise HTTPException(status_code=503, detail="Model not loaded (preprocess-only mode)")
    with gpu_lock:
        r = p.answer_query(req.query, top_k=req.top_k, subject=req.subject, class_num=req.class_num,
                           language=req.language, max_new_tokens=req.max_new_tokens)
    intent = r.get("intent") or {}
    return {
        "query": req.query,
        "answer": r["generation"]["hinglish_answer"],
        "language": req.language,
        "intent": intent.get("intent"),
        "intent_confidence": intent.get("confidence"),
        "intent_used_in_prompt": r["generation"]["prompt_intent"],
        "sources": r["retrieval"],
        "timings": {t["step"]: t["time_sec"] for t in r["pipeline_trace"]},
    }


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

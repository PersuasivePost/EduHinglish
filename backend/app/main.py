"""
EduHinglish backend API — what the frontend calls.

It holds no model. Every question is forwarded to the model server (notebook 07
on Colab, or src/model_server.py on a GPU machine) at MODEL_SERVER_URL.

Run (from the repo root):
    pip install -r backend/requirements.txt
    cp backend/.env.example backend/.env      # then fill in MODEL_SERVER_URL + MODEL_API_KEY
    uvicorn backend.app.main:app --reload --port 8080
Docs: http://localhost:8080/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .model_client import ModelClient, ModelServerError
from .schemas import AskRequest, AskResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelClient()
    yield
    await app.state.model.close()


app = FastAPI(title="EduHinglish Backend", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


def _raise(e: ModelServerError):
    raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    try:
        model = await app.state.model.health()
    except ModelServerError as e:
        model = {"status": "offline", "detail": e.detail}
    return HealthResponse(model_server_url=config.MODEL_SERVER_URL or "(not set)", model_server=model)


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

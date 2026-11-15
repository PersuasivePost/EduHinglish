"""Streaming tests with fake models (no GPU). Run: python tests/test_streaming.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")

from fastapi.testclient import TestClient  # noqa: E402


def sse_events(response) -> list[dict]:
    return [json.loads(l[6:]) for l in response.text.splitlines() if l.startswith("data: ")]


class FakePipeline:
    mode = "full"

    def answer_query(self, query, on_event=None, **kw):
        on_event({"type": "status", "stage": "retrieve"})
        on_event({"type": "meta", "standalone_query": query, "intent": "definition", "sources": []})
        for t in ["F = ", "ma"]:
            on_event({"type": "token", "text": t})
        return {"query": query, "standalone_query": query, "intent": {"intent": "definition", "confidence": 0.9},
                "retrieval": [], "generation": {"hinglish_answer": "F = ma", "prompt_intent": None},
                "pipeline_trace": [{"step": "total", "time_sec": 1.0}]}


def test_model_server_stream():
    import model_server
    model_server.state["pipeline"] = FakePipeline()
    client = TestClient(model_server.app)          # no lifespan: pipeline already set
    r = client.post("/ask/stream", json={"query": "Newton ka law?", "history": []})
    assert r.headers["content-type"].startswith("text/event-stream")
    kinds = [e["type"] for e in sse_events(r)]
    assert kinds == ["status", "meta", "token", "token", "done"], kinds
    assert sse_events(r)[-1]["answer"] == "F = ma"


def test_model_server_stream_error_event():
    import model_server

    class Boom(FakePipeline):
        def answer_query(self, *a, **k):
            raise ValueError("Unknown subject 'x'")

    model_server.state["pipeline"] = Boom()
    ev = sse_events(TestClient(model_server.app).post("/ask/stream", json={"query": "hello there"}))
    assert ev[-1]["type"] == "error" and "Unknown subject" in ev[-1]["detail"]


def test_backend_stream_saves_only_on_done():
    from backend.app import db
    from backend.app.main import app
    from backend.app.model_client import ModelServerError

    class FakeModel:
        fail = False

        async def ask_stream(self, payload):
            if self.fail:
                raise ModelServerError(503, "Model server unreachable")
            yield {"type": "status", "stage": "generate"}
            yield {"type": "token", "text": "F = ma"}
            yield {"type": "done", "answer": "F = ma", "sources": [], "timings": {}, "standalone_query": "q"}

        async def close(self):
            pass

    with TestClient(app) as client:
        fake = FakeModel()
        app.state.model = fake
        fake.fail = True
        ev = sse_events(client.post("/api/chat/stream", json={"message": "Newton ka law?"}))
        assert ev == [{"type": "error", "status": 503, "detail": "Model server unreachable"}]
        assert db.list_chats() == []                              # nothing saved on failure
        fake.fail = False
        ev = sse_events(client.post("/api/chat/stream", json={"message": "Newton ka law?"}))
        assert [e["type"] for e in ev] == ["status", "token", "done"]
        assert ev[-1]["assistant_message"]["content"] == "F = ma" and ev[-1]["chat"]["title"] == "Newton ka law?"
        assert len(db.list_chats()) == 1
        assert client.post("/api/chat/stream", json={"message": "hello", "chat_id": "nope"}).status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

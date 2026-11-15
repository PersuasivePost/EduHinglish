"""Follow-up / chat-context tests. Run: python tests/test_followup.py  (pytest also works)."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")

from followup import (build_rewrite_prompt, clean_rewrite, heuristic_rewrite, is_followup,  # noqa: E402
                      trim_history)

HIST = [{"role": "user", "content": "Newton ka second law kya hai?"},
        {"role": "assistant", "content": "Force = mass x acceleration. " * 30}]


def test_no_history_is_never_followup():
    assert not is_followup("aur iska formula?", [])


def test_followup_detection():
    assert is_followup("Kyun hota hai aisa?", HIST)               # no cue words needed
    assert is_followup("What about the third law?", HIST)
    assert not is_followup("Newton ka third law kya hai?", [])     # first message: nothing to resolve


def test_trim_history():
    long = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    assert len(trim_history(long)) == 6
    t = trim_history(HIST)
    assert t[1]["content"].endswith("...") and len(t[1]["content"]) <= 304
    assert trim_history([{"role": "user", "content": "  "}, {"role": "system", "content": "x"}]) == []


def test_rewrite_prompt_contains_context():
    p = build_rewrite_prompt(HIST, "aur iska formula?")
    assert "Student: Newton ka second law kya hai?" in p and "Last question: aur iska formula?" in p
    assert p.rstrip().endswith("Standalone question:")


def test_clean_rewrite():
    assert clean_rewrite('Standalone question: "Newton ke second law ka formula kya hai?"\nextra') == \
        "Newton ke second law ka formula kya hai?"
    assert clean_rewrite("") is None
    assert clean_rewrite("x" * 500) is None
    assert clean_rewrite("Force equals mass times acceleration, which means " * 4, "iska formula?") is None  # an answer


def test_heuristic_rewrite():
    assert heuristic_rewrite("iska formula?", HIST) == "Newton ka second law kya hai? iska formula?"
    assert heuristic_rewrite("iska formula?", []) == "iska formula?"


def test_backend_sends_history_on_second_turn():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    class FakeModel:
        def __init__(self):
            self.calls = []

        async def ask(self, payload):
            self.calls.append(payload)
            return {"answer": f"answer to: {payload['query']}", "sources": [], "timings": {},
                    "standalone_query": payload["query"]}

        async def close(self):
            pass

    with TestClient(app) as client:
        fake = FakeModel()
        app.state.model = fake
        r1 = client.post("/api/chat", json={"message": "Newton ka second law kya hai?"})
        chat_id = r1.json()["chat"]["id"]
        assert fake.calls[0]["history"] == []
        r2 = client.post("/api/chat", json={"message": "aur iska formula?", "chat_id": chat_id})
        assert r2.status_code == 200
        assert fake.calls[1]["history"] == [
            {"role": "user", "content": "Newton ka second law kya hai?"},
            {"role": "assistant", "content": "answer to: Newton ka second law kya hai?"}]
        assert r2.json()["standalone_query"] == "aur iska formula?"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

"""SQLite storage for chats, messages and 👍/👎 feedback (stdlib only, one file)."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    subject     TEXT,
    class_num   TEXT,
    language    TEXT NOT NULL DEFAULT 'hinglish',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id                 TEXT PRIMARY KEY,
    chat_id            TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role               TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content            TEXT NOT NULL,
    intent             TEXT,
    intent_confidence  REAL,
    sources            TEXT NOT NULL DEFAULT '[]',
    timings            TEXT NOT NULL DEFAULT '{}',
    feedback           INTEGER NOT NULL DEFAULT 0,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
"""


def init():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript(SCHEMA)


@contextmanager
def connect():
    db = sqlite3.connect(config.DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
        db.commit()
    finally:
        db.close()


def _chat(row) -> dict:
    return dict(row)


def _message(row) -> dict:
    m = dict(row)
    m["sources"] = json.loads(m["sources"])
    m["timings"] = json.loads(m["timings"])
    return m


# ── Chats ────────────────────────────────────────────────────────────────────

def create_chat(title: str, subject=None, class_num=None, language="hinglish") -> dict:
    now = time.time()
    chat = {"id": uuid.uuid4().hex, "title": title, "subject": subject, "class_num": class_num,
            "language": language, "created_at": now, "updated_at": now}
    with connect() as db:
        db.execute("INSERT INTO chats VALUES (:id, :title, :subject, :class_num, :language, :created_at, :updated_at)",
                   chat)
    return chat


def list_chats() -> list[dict]:
    with connect() as db:
        return [_chat(r) for r in db.execute("SELECT * FROM chats ORDER BY updated_at DESC")]


def get_chat(chat_id: str):
    with connect() as db:
        row = db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return _chat(row) if row else None


def update_chat(chat_id: str, **fields) -> dict | None:
    fields = {k: v for k, v in fields.items() if k in {"title", "subject", "class_num", "language"}}
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with connect() as db:
        db.execute(f"UPDATE chats SET {sets} WHERE id = :id", {**fields, "id": chat_id})
    return get_chat(chat_id)


def delete_chat(chat_id: str) -> bool:
    with connect() as db:
        return db.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount > 0


# ── Messages ─────────────────────────────────────────────────────────────────

def add_message(chat_id: str, role: str, content: str, intent=None, intent_confidence=None,
                sources=None, timings=None) -> dict:
    row = {"id": uuid.uuid4().hex, "chat_id": chat_id, "role": role, "content": content,
           "intent": intent, "intent_confidence": intent_confidence,
           "sources": json.dumps(sources or [], ensure_ascii=False),
           "timings": json.dumps(timings or {}), "feedback": 0, "created_at": time.time()}
    with connect() as db:
        db.execute("INSERT INTO messages VALUES (:id, :chat_id, :role, :content, :intent, :intent_confidence,"
                   " :sources, :timings, :feedback, :created_at)", row)
        db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (row["created_at"], chat_id))
    return {**row, "sources": sources or [], "timings": timings or {}}


def list_messages(chat_id: str) -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at", (chat_id,))
        return [_message(r) for r in rows]


def set_feedback(message_id: str, value: int) -> dict | None:
    with connect() as db:
        db.execute("UPDATE messages SET feedback = ? WHERE id = ? AND role = 'assistant'", (value, message_id))
        row = db.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _message(row) if row else None

"""Storage for chats, messages and 👍/👎 feedback.

DATABASE_URL set (e.g. a Neon connection string)  -> PostgreSQL via psycopg.
Not set                                            -> a local SQLite file (DB_PATH), zero setup.
Every query below is written once with :name / ? placeholders; _pg_sql() adapts it for Postgres.
"""

import json
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    google_sub  TEXT NOT NULL UNIQUE,
    email       TEXT,
    name        TEXT,
    picture     TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    subject     TEXT,
    class_num   TEXT,
    language    TEXT NOT NULL DEFAULT 'hinglish',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    user_id     TEXT,
    share_token TEXT,
    share_upto  REAL
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


USE_PG = bool(config.DATABASE_URL)

# Same tables on Postgres: timestamps are float seconds, so REAL (4 bytes) would lose precision.
PG_SCHEMA = SCHEMA.replace("REAL", "DOUBLE PRECISION")


def _pg_sql(sql: str) -> str:
    """':name' -> '%(name)s' and '?' -> '%s' (psycopg's placeholder styles)."""
    return re.sub(r":(\w+)", r"%(\1)s", sql).replace("?", "%s")


class _PgConn:
    """Gives a psycopg connection the sqlite-style execute(sql, params) the queries below use."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=None):
        return self._conn.execute(_pg_sql(sql), params)


# Columns added after the first release; older databases get them on startup.
_NEW_CHAT_COLUMNS = {"user_id": "TEXT", "share_token": "TEXT", "share_upto": "REAL"}


def init():
    if USE_PG:
        with connect() as db:
            for stmt in filter(None, (s.strip() for s in PG_SCHEMA.split(";"))):
                db.execute(stmt)
            for col, typ in _NEW_CHAT_COLUMNS.items():
                db.execute(f"ALTER TABLE chats ADD COLUMN IF NOT EXISTS {col} "
                           f"{'DOUBLE PRECISION' if typ == 'REAL' else typ}")
            _create_indexes(db)
        return
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript(SCHEMA)
        have = {r["name"] for r in db.execute("PRAGMA table_info(chats)")}
        for col, typ in _NEW_CHAT_COLUMNS.items():
            if col not in have:
                db.execute(f"ALTER TABLE chats ADD COLUMN {col} {typ}")
        _create_indexes(db)


def _create_indexes(db):
    db.execute("CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_chats_share ON chats(share_token)")


@contextmanager
def connect():
    if USE_PG:
        import psycopg
        from psycopg.rows import dict_row

        # prepare_threshold=None: Neon's pooled endpoint (pgbouncer) does not support prepared statements
        conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row, prepare_threshold=None)
        try:
            yield _PgConn(conn)
            conn.commit()
        finally:
            conn.close()
        return
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


# ── Users ────────────────────────────────────────────────────────────────────

def upsert_user(google_sub: str, email=None, name=None, picture=None) -> dict:
    """Create the user on first Google sign-in, refresh name/picture on later ones."""
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE google_sub = ?", (google_sub,)).fetchone()
        if row:
            db.execute("UPDATE users SET email = ?, name = ?, picture = ? WHERE id = ?",
                       (email, name, picture, row["id"]))
            uid = row["id"]
        else:
            uid = uuid.uuid4().hex
            db.execute("INSERT INTO users (id, google_sub, email, name, picture, created_at)"
                       " VALUES (?, ?, ?, ?, ?, ?)", (uid, google_sub, email, name, picture, time.time()))
        return dict(db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone())


def get_user(user_id: str):
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# ── Chats (always scoped to the owner's user_id) ─────────────────────────────

_PUBLIC_CHAT_COLS = "id, title, subject, class_num, language, created_at, updated_at, share_token"


def create_chat(user_id: str, title: str, subject=None, class_num=None, language="hinglish") -> dict:
    now = time.time()
    chat = {"id": uuid.uuid4().hex, "title": title, "subject": subject, "class_num": class_num,
            "language": language, "created_at": now, "updated_at": now, "user_id": user_id}
    with connect() as db:
        db.execute("INSERT INTO chats (id, title, subject, class_num, language, created_at, updated_at, user_id)"
                   " VALUES (:id, :title, :subject, :class_num, :language, :created_at, :updated_at, :user_id)",
                   chat)
    return get_chat(chat["id"], user_id)


def list_chats(user_id: str) -> list[dict]:
    with connect() as db:
        return [_chat(r) for r in db.execute(
            f"SELECT {_PUBLIC_CHAT_COLS} FROM chats WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))]


def get_chat(chat_id: str, user_id: str):
    """None when the chat does not exist OR belongs to someone else (same answer, nothing leaks)."""
    with connect() as db:
        row = db.execute(f"SELECT {_PUBLIC_CHAT_COLS} FROM chats WHERE id = ? AND user_id = ?",
                         (chat_id, user_id)).fetchone()
        return _chat(row) if row else None


def update_chat(chat_id: str, user_id: str, **fields) -> dict | None:
    fields = {k: v for k, v in fields.items() if k in {"title", "subject", "class_num", "language"}}
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with connect() as db:
        db.execute(f"UPDATE chats SET {sets} WHERE id = :id AND user_id = :user_id",
                   {**fields, "id": chat_id, "user_id": user_id})
    return get_chat(chat_id, user_id)


def delete_chat(chat_id: str, user_id: str) -> bool:
    with connect() as db:
        return db.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).rowcount > 0


# ── Sharing: a read-only snapshot at an unguessable link ─────────────────────

def share_chat(chat_id: str, user_id: str):
    """Publish the chat as it is now. Messages sent later are not visible until shared again."""
    chat = get_chat(chat_id, user_id)
    if not chat:
        return None
    token = chat.get("share_token") or secrets.token_urlsafe(16)
    with connect() as db:
        db.execute("UPDATE chats SET share_token = ?, share_upto = ? WHERE id = ? AND user_id = ?",
                   (token, time.time(), chat_id, user_id))
    return get_chat(chat_id, user_id)


def unshare_chat(chat_id: str, user_id: str) -> bool:
    with connect() as db:
        return db.execute("UPDATE chats SET share_token = NULL, share_upto = NULL WHERE id = ? AND user_id = ?",
                          (chat_id, user_id)).rowcount > 0


def get_shared(token: str):
    """(title, messages) for a share link, or None. Needs no login and no model server."""
    if not token:
        return None
    with connect() as db:
        chat = db.execute("SELECT id, title, subject, share_upto FROM chats WHERE share_token = ?",
                          (token,)).fetchone()
        if not chat:
            return None
        rows = db.execute("SELECT * FROM messages WHERE chat_id = ? AND created_at <= ? ORDER BY created_at",
                          (chat["id"], chat["share_upto"] or 0))
        return dict(chat), [_message(r) for r in rows]


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


def set_feedback(message_id: str, user_id: str, value: int) -> dict | None:
    """Only the chat's owner can rate its messages."""
    with connect() as db:
        db.execute("UPDATE messages SET feedback = ? WHERE id = ? AND role = 'assistant' AND chat_id IN"
                   " (SELECT id FROM chats WHERE user_id = ?)", (value, message_id, user_id))
        row = db.execute("SELECT * FROM messages WHERE id = ? AND chat_id IN"
                         " (SELECT id FROM chats WHERE user_id = ?)", (message_id, user_id)).fetchone()
        return _message(row) if row else None

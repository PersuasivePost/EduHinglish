"""Chat storage test, runs against SQLite by default or Postgres when DATABASE_URL is set.
Run: python tests/test_db.py"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DB_PATH", str(Path(tempfile.mkdtemp()) / "test.db"))

from backend.app import db  # noqa: E402


def test_pg_placeholder_translation():
    assert db._pg_sql("UPDATE t SET a = :a WHERE id = ?") == "UPDATE t SET a = %(a)s WHERE id = %s"


def test_chat_roundtrip():
    db.init()
    db.init()   # idempotent
    chat = db.create_chat("Newton", subject="physics", class_num="9")
    assert db.get_chat(chat["id"])["subject"] == "physics"
    u = db.add_message(chat["id"], "user", "Newton ka second law kya hai?")
    a = db.add_message(chat["id"], "assistant", "F = ma", intent="definition", intent_confidence=0.9,
                       sources=[{"chapter": "Force"}], timings={"total": 1.5})
    msgs = db.list_messages(chat["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"] == [{"chapter": "Force"}] and msgs[1]["timings"] == {"total": 1.5}
    assert abs(msgs[0]["created_at"] - u["created_at"]) < 1e-3   # float precision kept
    assert db.set_feedback(a["id"], 1)["feedback"] == 1
    assert db.set_feedback(u["id"], 1)["feedback"] == 0           # feedback only on assistant messages
    assert db.update_chat(chat["id"], title="Renamed", language="english")["title"] == "Renamed"
    assert db.list_chats()[0]["id"] == chat["id"]
    assert db.delete_chat(chat["id"]) is True
    assert db.list_messages(chat["id"]) == []                     # cascade delete
    assert db.delete_chat(chat["id"]) is False


if __name__ == "__main__":
    print("backend:", "postgres" if db.USE_PG else "sqlite")
    test_pg_placeholder_translation()
    test_chat_roundtrip()
    print("PASS")

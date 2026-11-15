"""Backend settings, read from environment variables or backend/.env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Public URL printed by notebook 07, e.g. https://xxxx.trycloudflare.com or https://name.ngrok-free.app
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "").rstrip("/")
# Same value as the MODEL_API_KEY secret in Colab
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "")
# Gemma on a T4 takes ~10-40 s per answer; allow for a queue of a few requests
MODEL_TIMEOUT_SEC = float(os.environ.get("MODEL_TIMEOUT_SEC", "180"))
# Frontend origins allowed to call the backend (comma-separated; * = any)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
# PostgreSQL connection string (Neon/Supabase). When set it is used instead of the SQLite file below.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
# SQLite file for chats, messages and feedback
DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "eduhinglish.db"))
# How many earlier messages of a chat are sent to the model for follow-up questions
HISTORY_MESSAGES = int(os.environ.get("HISTORY_MESSAGES", "6"))
# Google sign-in. Comma-separated OAuth client ids (web + Android/iOS + the server/web one the app uses).
# Empty = sign-in off: everyone is one local user (fine for local testing, NOT for the public deployment).
GOOGLE_CLIENT_IDS = [c.strip() for c in os.environ.get("GOOGLE_CLIENT_IDS", "").split(",") if c.strip()]
# Signs our session tokens. Set a long random value on Render: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-change-me")
SESSION_DAYS = int(os.environ.get("SESSION_DAYS", "30"))

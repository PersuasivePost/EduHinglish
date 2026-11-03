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

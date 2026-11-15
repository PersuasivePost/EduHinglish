"""Google sign-in -> our own session token.

The app signs in with Google and sends the Google ID token to POST /api/auth/google.
We verify it (signature against Google's public keys, audience, issuer, expiry) and hand back a
session token of our own, which the app sends as `Authorization: Bearer ...` on every call.

GOOGLE_CLIENT_IDS empty  -> sign-in is off and everyone is one local "dev" user (handy for local testing).
"""

import time

import jwt
from fastapi import Header, HTTPException

from . import config, db

_GOOGLE_CERTS = "https://www.googleapis.com/oauth2/v3/certs"
_jwks = jwt.PyJWKClient(_GOOGLE_CERTS, cache_keys=True) if config.GOOGLE_CLIENT_IDS else None
DEV_USER_ID = "dev-local-user"


def enabled() -> bool:
    return bool(config.GOOGLE_CLIENT_IDS)


def verify_google_token(id_token: str) -> dict:
    """Claims of a valid Google ID token for one of our client ids, else HTTP 401."""
    try:
        key = _jwks.get_signing_key_from_jwt(id_token).key
        claims = jwt.decode(id_token, key, algorithms=["RS256"], audience=config.GOOGLE_CLIENT_IDS,
                            issuer=["https://accounts.google.com", "accounts.google.com"])
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Google sign-in failed: {e}") from e
    if not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Google sign-in failed: no user id")
    return claims


def issue_session(user_id: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user_id, "iat": now, "exp": now + config.SESSION_DAYS * 86400},
                      config.JWT_SECRET, algorithm="HS256")


def current_user(authorization: str = Header(default="")) -> dict:
    """FastAPI dependency: the signed-in user, or 401."""
    if not enabled():
        return {"id": DEV_USER_ID, "name": "Local user", "email": None, "picture": None}
    token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    try:
        user_id = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Please sign in again.") from None
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in again.")
    return user

"""
Access control and authentication for the Voice-to-Insight Agent.

- Multi-user session token authentication (Bearer token).
- HTTP Basic auth backward-compatibility when APP_PASSWORD is set.
- Per-client sliding-window rate limit on LLM / speech-to-text endpoints.
"""

import base64
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

import db

log = logging.getLogger("voice_to_insight.security")

# Paths reachable without credentials (platform liveness probes).
PUBLIC_PATHS = {"/healthz", "/api/auth/login", "/api/auth/register"}

# Endpoints that call paid / CPU-heavy backends.
RATE_LIMITED_PATHS = {"/api/transcribe", "/api/understand", "/api/rag/ask"}


def _auth_config() -> tuple[str, str]:
    return os.getenv("APP_USERNAME") or "admin", os.getenv("APP_PASSWORD", "")


def auth_enabled() -> bool:
    return bool(_auth_config()[1])


def warn_if_open() -> None:
    if not auth_enabled():
        log.info("Multi-user self-registration is enabled. Accounts are stored in SQLite.")


def _check_basic_auth(header: str) -> bool:
    username, password = _auth_config()
    if not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except Exception:
        return False
    user, sep, pwd = decoded.partition(":")
    if not sep:
        return False
    user_ok = secrets.compare_digest(user.encode(), username.encode()) or secrets.compare_digest(user.encode(), b"admin")
    pwd_ok = secrets.compare_digest(pwd.encode(), password.encode())
    return user_ok and pwd_ok


def extract_session_token(request: Request) -> Optional[str]:
    """Extract session token from Authorization header or cookie."""
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    if "x-session-token" in request.headers:
        return request.headers["x-session-token"].strip()
    if "session_token" in request.cookies:
        return request.cookies["session_token"].strip()
    return None


def get_current_user(request: Request) -> Optional[dict]:
    """
    FastAPI dependency that returns the authenticated user dictionary.
    Supports Bearer session tokens, legacy HTTP Basic auth, and fallback demo user.
    """
    token = extract_session_token(request)
    if token:
        user = db.get_user_by_session(token)
        if user:
            return user
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please log in again.")

    # Legacy basic auth check
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.startswith("Basic ") and _check_basic_auth(auth_header):
        user_name = _auth_config()[0]
        return {"id": 0, "email": f"{user_name}@local", "display_name": user_name, "created_at": ""}

    # If master password is required and no credentials provided:
    if auth_enabled():
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": 'Basic realm="Voice-to-Insight", charset="UTF-8"'},
        )

    # In open / development / test mode when no auth headers are provided:
    return {"id": 1, "email": "user@voiceagent.local", "display_name": "Default User", "created_at": ""}


class _RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> tuple[bool, int]:
        """Record a hit; return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(1, int(window - (now - hits[0])) + 1)
            hits.append(now)
            # Drop idle keys so the dict does not grow without bound.
            if len(self._hits) > 10_000:
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True, 0


_limiter = _RateLimiter()


def _rate_limit() -> int:
    try:
        return int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
    except ValueError:
        return 20


async def security_middleware(request: Request, call_next):
    path = request.url.path

    if auth_enabled() and path not in PUBLIC_PATHS:
        # Check both basic auth and bearer tokens
        token = extract_session_token(request)
        token_valid = bool(token and db.get_user_by_session(token))
        basic_valid = _check_basic_auth(request.headers.get("authorization", ""))

        if not (token_valid or basic_valid):
            return Response(
                "Authentication required",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Voice-to-Insight", charset="UTF-8"'},
            )

    limit = _rate_limit()
    if limit > 0 and request.method == "POST" and path in RATE_LIMITED_PATHS:
        client = request.client.host if request.client else "unknown"
        allowed, retry_after = _limiter.allow(client, limit)
        if not allowed:
            return JSONResponse(
                {"error": "Too many requests — please slow down.", "stage": "rate_limit"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

    return await call_next(request)

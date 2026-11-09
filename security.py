"""
Access control for the Voice-to-Insight Agent.

- HTTP Basic auth over the whole app (UI + API) when APP_PASSWORD is set.
  Browsers prompt once and then send the credentials with every same-origin
  fetch, so the frontend needs no changes.
- A per-client, in-memory sliding-window rate limit on the endpoints that
  spend LLM / speech-to-text quota.

The in-memory limiter is per process: with several workers or instances, each
keeps its own counters. Put a shared limiter (e.g. at the reverse proxy) in
front if you scale out.
"""

import base64
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse, Response

log = logging.getLogger("voice_to_insight.security")

# Paths reachable without credentials (platform liveness probes).
PUBLIC_PATHS = {"/healthz"}

# Endpoints that call paid / CPU-heavy backends.
RATE_LIMITED_PATHS = {"/api/transcribe", "/api/understand", "/api/rag/ask"}


def _auth_config() -> tuple[str, str]:
    return os.getenv("APP_USERNAME", "admin"), os.getenv("APP_PASSWORD", "")


def auth_enabled() -> bool:
    return bool(_auth_config()[1])


def warn_if_open() -> None:
    if not auth_enabled():
        log.warning(
            "APP_PASSWORD is not set: the app and its API are open to anyone who can reach it. "
            "Set APP_PASSWORD before deploying."
        )


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
    user_ok = secrets.compare_digest(user.encode(), username.encode())
    pwd_ok = secrets.compare_digest(pwd.encode(), password.encode())
    return user_ok and pwd_ok


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
        if not _check_basic_auth(request.headers.get("authorization", "")):
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

"""
Public demo mode (DEMO_MODE=true).

Lets you share a live link (e.g. on LinkedIn) that runs on *your* Gemini key while
capping what strangers can spend:

- visitors must sign in (free account), so each person gets a private workspace;
- every request that calls the language model is counted per account, per IP
  address and in total per day (UTC), and refused once a cap is reached;
- voice transcriptions (server CPU) have their own per-account daily cap;
- sign-ups are capped per IP per day;
- notes, tasks, captions and summaries are deleted after DEMO_RETENTION_HOURS;
- Notion sync is off (strangers must not write to your Notion);
- DEMO_CLOSED=true ends the demo without taking the site down.

Counters live in SQLite, so they survive restarts. Set any cap to 0 to block it.
"""

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Request

import db

CLOSED_MESSAGE = "This demo has ended. Thanks for trying it!"


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def enabled() -> bool:
    return _flag("DEMO_MODE")


def closed() -> bool:
    return _flag("DEMO_CLOSED")


def retention_hours() -> int:
    return _int("DEMO_RETENTION_HOURS", 24)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _seconds_until_reset() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds()) + 1


def client_ip(request: Request) -> str:
    # Behind a proxy, run uvicorn with --proxy-headers so this is the real client IP.
    return request.client.host if request.client else "unknown"


def _llm_checks(request: Request, user: dict) -> list[tuple[str, int]]:
    return [
        (f"llm:user:{user.get('id')}", _int("DEMO_DAILY_PER_USER", 10)),
        (f"llm:ip:{client_ip(request)}", _int("DEMO_DAILY_PER_IP", 30)),
        ("llm:total", _int("DEMO_DAILY_TOTAL", 200)),
    ]


_MESSAGES = {
    "llm:user": "You've used today's free demo requests. Come back tomorrow!",
    "llm:ip": "Today's free demo requests from your network are used up. Come back tomorrow!",
    "llm:total": "The demo has reached today's overall limit. Please come back tomorrow!",
    "stt:user": "You've used today's free voice transcriptions. Come back tomorrow, or type your note instead.",
    "signup:ip": "Too many new accounts from your network today. Please try again tomorrow.",
}


def _charge(checks: list[tuple[str, int]]) -> None:
    if closed():
        raise HTTPException(status_code=503, detail=CLOSED_MESSAGE)
    maybe_cleanup()
    allowed, full_key = db.consume_quota(_today(), checks)
    if not allowed:
        kind = ":".join(full_key.split(":")[:2]) if full_key else ""
        raise HTTPException(
            status_code=429,
            detail=_MESSAGES.get(kind, "Daily demo limit reached. Come back tomorrow!"),
            headers={"Retry-After": str(_seconds_until_reset())},
        )


def charge_llm(request: Request, user: dict) -> None:
    """Count one language-model request against the demo quotas, or raise 429/503."""
    if enabled():
        _charge(_llm_checks(request, user))


def charge_transcription(request: Request, user: dict) -> None:
    if enabled():
        _charge([(f"stt:user:{user.get('id')}", _int("DEMO_TRANSCRIBE_DAILY_PER_USER", 20))])


def charge_signup(request: Request) -> None:
    if enabled():
        _charge([(f"signup:ip:{client_ip(request)}", _int("DEMO_SIGNUPS_PER_IP", 3))])


def status(request: Request, user: Optional[dict]) -> dict:
    if not enabled():
        return {"demo": False}
    info = {
        "demo": True,
        "closed": closed(),
        "retention_hours": retention_hours(),
        "daily_limit": _int("DEMO_DAILY_PER_USER", 10),
    }
    if user:
        info["remaining_today"] = 0 if closed() else db.quota_remaining(_today(), _llm_checks(request, user))
    return info


# --- Retention cleanup (runs at most every few minutes, piggybacking on requests) ---
_last_cleanup = 0.0
_cleanup_lock = threading.Lock()
CLEANUP_INTERVAL_SECONDS = 300


def maybe_cleanup(force: bool = False) -> None:
    global _last_cleanup
    if not enabled():
        return
    now = time.monotonic()
    with _cleanup_lock:
        if not force and now - _last_cleanup < CLEANUP_INTERVAL_SECONDS:
            return
        _last_cleanup = now
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=retention_hours())).isoformat(timespec="seconds")
    keep_since = (datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat()
    db.purge_older_than(cutoff, keep_since)

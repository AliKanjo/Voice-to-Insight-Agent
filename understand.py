"""
Understanding layer: transcript -> intent classification + structured extraction.

Supports multilingual voice notes (Arabic, English, French, Spanish, German, Turkish, etc., or mixed).
Classifies into:
  - "task": action item / to-do
  - "content": social media caption
  - "summary": structured summary
  - "query": question asking about past recorded notes/tasks (RAG)

Interchangeable backends: "openai", "gemini", or "local" (Ollama / llama.cpp).
"""

import json
import os
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

from openai import OpenAI

import arabic_norm

SYSTEM_PROMPT = """You are an intelligent multilingual assistant that converts a spoken voice note into a single structured record.
The user may speak in ANY language (Arabic, English, French, Spanish, German, Turkish, Russian, etc.) or a code-switched mix.

Classify the note into exactly ONE kind:
- "task": the speaker wants something done / a to-do / a reminder / an action item.
- "content": the speaker wants a social media caption written (Instagram or LinkedIn).
- "summary": the speaker is dictating thoughts or meeting notes they want summarized.
- "query": the speaker is asking a question about their past notes, tasks, or information (e.g. "What did I do yesterday?", "Show me high priority tasks", "ما هي المهام المطلوبة مني؟").

Rules:
- Always respond in the SAME language the note is mostly in (Arabic in -> Arabic out, French in -> French out, etc.).
- For tasks: infer a concise title.
  owner is the person who must DO the task (the assignee). It is NOT the recipient, the person being
  contacted, or the subject of the task. "Send the invoice to Sara" -> owner null (Sara is the recipient).
  "Ask Omar to fix the website" -> owner "Omar". If the speaker does it themself or nobody is named, owner is null.
  due_date is the deadline as spoken (e.g. "tomorrow morning", "next Friday", "غدا", "demain"), or null if none.
  due_date_iso is that deadline resolved to a calendar date "YYYY-MM-DD" using the current date given below
  (for "before Friday" use that Friday), or null if there is no deadline or it cannot be pinned to a day.
  priority is one of "low", "medium", "high" (default "medium" if unclear).
- For content: pick the most fitting platform, write a ready-to-post caption, and name the tone.
- For summary: write a concise summary_text and 2-5 key_points.
- For query: return the clean formulated question to search for in past notes.
"""

_TASK_BRANCH = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "title", "owner", "due_date", "due_date_iso", "priority"],
    "properties": {
        "kind": {"type": "string", "enum": ["task"]},
        "title": {"type": "string"},
        "owner": {"type": ["string", "null"]},
        "due_date": {"type": ["string", "null"]},
        "due_date_iso": {"type": ["string", "null"]},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

_CONTENT_BRANCH = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "platform", "caption", "tone"],
    "properties": {
        "kind": {"type": "string", "enum": ["content"]},
        "platform": {"type": "string", "enum": ["instagram", "linkedin"]},
        "caption": {"type": "string"},
        "tone": {"type": "string"},
    },
}

_SUMMARY_BRANCH = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "summary_text", "key_points"],
    "properties": {
        "kind": {"type": "string", "enum": ["summary"]},
        "summary_text": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
}

_QUERY_BRANCH = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "question"],
    "properties": {
        "kind": {"type": "string", "enum": ["query"]},
        "question": {"type": "string"},
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["result"],
    "properties": {
        "result": {"anyOf": [_TASK_BRANCH, _CONTENT_BRANCH, _SUMMARY_BRANCH, _QUERY_BRANCH]},
    },
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "voice_insight",
        "strict": True,
        "schema": RESPONSE_SCHEMA,
    },
}


def _timeout() -> float:
    """Per-request timeout for LLM calls (the SDK default is 10 minutes)."""
    try:
        return float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    except ValueError:
        return 60.0


def make_client(**kwargs) -> OpenAI:
    # One SDK retry on top of our own Gemini fallback keeps worst-case latency bounded.
    return OpenAI(timeout=_timeout(), max_retries=1, **kwargs)


def today() -> date:
    """Current date in APP_TIMEZONE (default UTC), used to resolve relative deadlines."""
    try:
        tz = ZoneInfo(os.getenv("APP_TIMEZONE", "UTC"))
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).date()


def _system_prompt() -> str:
    d = today()
    return f"{SYSTEM_PROMPT}\nCurrent date: {d.isoformat()} ({d.strftime('%A')}).\n"


def _valid_iso_date(value) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def active_backend() -> str:
    return os.getenv("UNDERSTAND_BACKEND", "openai").strip().lower()


def _resolve_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) for the selected backend."""
    backend = active_backend()
    if backend == "local":
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080/v1")
        model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-1.5b-instruct")
        return make_client(base_url=base_url, api_key="sk-local-noop"), model

    if backend == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set — add it to your .env file.")
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        return make_client(base_url=base_url, api_key=key), model

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file.")
    return make_client(api_key=key), "gpt-4o"


def _is_transient(err: Exception) -> bool:
    msg = str(err)
    return any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


def _fail(error: str, error_type: str = "upstream", **extra) -> dict:
    """Error result. error_type is "bad_input" (caller's fault, safe to show)
    or "upstream" (backend/model failure; details are logged, not shown, by main.py)."""
    return {"intent": "error", "error": error, "error_type": error_type, **extra}


def understand(transcript: str) -> dict:
    """Return {"intent": ..., "data": {...}} or {"intent": "error", "error": ..., "error_type": ...}."""
    text = (transcript or "").strip()
    if not text:
        return _fail("Empty transcript — nothing to understand.", "bad_input")

    # Pre-clean transcript (strips diacritics, eastern numerals, tatweel)
    cleaned_text = arabic_norm.clean_transcript(text)
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": cleaned_text},
    ]

    try:
        client, model = _resolve_client()

        def _call(model_name: str):
            return client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                response_format=RESPONSE_FORMAT,
                messages=messages,
            )

        try:
            resp = _call(model)
        except Exception as api_err:
            # Gemini's free tier rate-limits / overloads often: back off once, then retry
            # on GEMINI_FALLBACK_MODEL if set, otherwise on the same model.
            if active_backend() != "gemini" or not _is_transient(api_err):
                raise
            time.sleep(5)
            resp = _call(os.getenv("GEMINI_FALLBACK_MODEL", "").strip() or model)
    except Exception as exc:
        hint = ""
        if active_backend() == "local":
            hint = " (is the local LLM server running at LOCAL_LLM_BASE_URL?)"
        return _fail(f"[{active_backend()}] request failed: {exc}{hint}")

    choice = resp.choices[0]
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        return _fail(f"Model refused: {refusal}")

    raw = choice.message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _fail(f"Could not parse model JSON: {exc}", raw=raw)

    result = parsed.get("result") if isinstance(parsed, dict) else None
    if not isinstance(result, dict):
        return _fail("Model returned an unexpected shape.", raw=parsed)

    intent = result.get("kind")
    if intent not in ("task", "content", "summary", "query"):
        return _fail(f"Unknown kind: {intent!r}", raw=parsed)

    if intent == "task":
        # Never trust the model's date format: keep only real YYYY-MM-DD dates.
        result["due_date_iso"] = _valid_iso_date(result.get("due_date_iso"))

    return {"intent": intent, "data": result}

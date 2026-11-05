"""
Understanding layer: transcript -> intent classification + structured extraction.

Two interchangeable "brains" behind the SAME strict-JSON contract, chosen at
startup via the UNDERSTAND_BACKEND env var:

  - "openai" (default): gpt-4o via the OpenAI API.
  - "local":            a small model (e.g. Qwen2.5-1.5B-Instruct) served by
                        llama.cpp's OpenAI-compatible server.

Both use grammar/schema-constrained decoding so the output is *forced* to be
valid, parseable JSON — even from a 0.5-1.5B model. Everything here is
defensive: any failure returns an `error` intent the UI renders gracefully.
"""

import json
import os

from openai import OpenAI

SYSTEM_PROMPT = """You are an assistant that turns a spoken voice note (which may be
in English, Arabic, or a code-switched mix of both) into a single structured record.

Classify the note into exactly one kind:
- "task": the speaker wants something done / a to-do / a reminder / an assignment.
- "content": the speaker wants a social media caption written (Instagram or LinkedIn).
- "summary": the speaker is dictating notes/thoughts they want summarized.

Then extract the fields for that kind. Rules:
- Always respond in the SAME language the note is mostly in (Arabic in -> Arabic out).
- For tasks: infer a concise title. owner/due_date are null if not mentioned.
  due_date should be a natural phrase as spoken (e.g. "tomorrow", "next Friday", "غدا").
  priority is one of low/medium/high (default medium if unclear).
- For content: pick the most fitting platform, write a ready-to-post caption, and
  name the tone (e.g. "professional", "playful", "inspirational").
- For summary: write a tight summary_text and 2-5 key_points.
"""

# --- Strict JSON schema for structured / grammar-constrained outputs --------
# A single `result` branch (anyOf) discriminated by `kind`. We derive the intent
# from `result.kind` rather than a separate top-level field, so there's no way
# for the model to pick a kind that disagrees with its data — one less failure
# mode for a small local model. Each branch is fully specified (all keys
# required, no extras); nullable fields use a ["string","null"] type.
_TASK_BRANCH = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "title", "owner", "due_date", "priority"],
    "properties": {
        "kind": {"type": "string", "enum": ["task"]},
        "title": {"type": "string"},
        "owner": {"type": ["string", "null"]},
        "due_date": {"type": ["string", "null"]},
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

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["result"],
    "properties": {
        "result": {"anyOf": [_TASK_BRANCH, _CONTENT_BRANCH, _SUMMARY_BRANCH]},
    },
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "voice_insight",
        "strict": True,          # honored by OpenAI; harmless/ignored by llama.cpp
        "schema": RESPONSE_SCHEMA,
    },
}


# --- Backend selection -----------------------------------------------------
def active_backend() -> str:
    return os.getenv("UNDERSTAND_BACKEND", "openai").strip().lower()


def _resolve_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) for the selected backend."""
    if active_backend() == "local":
        # llama.cpp's server is OpenAI-compatible; it ignores the API key but the
        # SDK requires a non-empty string. The model name is whatever the server
        # was launched with (llama-server accepts any name and ignores it).
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080/v1")
        model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-1.5b-instruct")
        return OpenAI(base_url=base_url, api_key="sk-local-noop"), model

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file.")
    return OpenAI(api_key=key), "gpt-4o"


def understand(transcript: str) -> dict:
    """Return {"intent": ..., "data": {...}} or {"intent": "error", "error": ...}."""
    text = (transcript or "").strip()
    if not text:
        return {"intent": "error", "error": "Empty transcript — nothing to understand."}

    try:
        client, model = _resolve_client()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format=RESPONSE_FORMAT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
    except Exception as exc:  # network / auth / server down — surface it loudly
        hint = ""
        if active_backend() == "local":
            hint = " (is the llama.cpp server running at LOCAL_LLM_BASE_URL?)"
        return {"intent": "error", "error": f"[{active_backend()}] request failed: {exc}{hint}"}

    choice = resp.choices[0]

    # Structured-output refusals come back on a dedicated field (OpenAI only).
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        return {"intent": "error", "error": f"Model refused: {refusal}"}

    raw = choice.message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"intent": "error", "error": f"Could not parse model JSON: {exc}", "raw": raw}

    result = parsed.get("result")
    if not isinstance(result, dict):
        return {"intent": "error", "error": "Model returned an unexpected shape.", "raw": parsed}

    intent = result.get("kind")
    if intent not in ("task", "content", "summary"):
        return {"intent": "error", "error": f"Unknown kind: {intent!r}", "raw": parsed}

    return {"intent": intent, "data": result}

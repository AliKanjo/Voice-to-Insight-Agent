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
- For tasks: infer a concise title. owner and due_date are null if not mentioned.
  due_date should be a natural phrase as spoken (e.g. "tomorrow", "next Friday", "غدا", "demain").
  priority is one of "low", "medium", "high" (default "medium" if unclear).
- For content: pick the most fitting platform, write a ready-to-post caption, and name the tone.
- For summary: write a concise summary_text and 2-5 key_points.
- For query: return the clean formulated question to search for in past notes.
"""

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


def active_backend() -> str:
    return os.getenv("UNDERSTAND_BACKEND", "openai").strip().lower()


def _resolve_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) for the selected backend."""
    backend = active_backend()
    if backend == "local":
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080/v1")
        model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-1.5b-instruct")
        return OpenAI(base_url=base_url, api_key="sk-local-noop"), model

    if backend == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set — add it to your .env file.")
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        return OpenAI(base_url=base_url, api_key=key), model

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file.")
    return OpenAI(api_key=key), "gpt-4o"


def understand(transcript: str) -> dict:
    """Return {"intent": ..., "data": {...}} or {"intent": "error", "error": ...}."""
    text = (transcript or "").strip()
    if not text:
        return {"intent": "error", "error": "Empty transcript — nothing to understand."}

    # Pre-clean transcript (strips diacritics, eastern numerals, tatweel)
    cleaned_text = arabic_norm.clean_transcript(text)

    try:
        client, model = _resolve_client()
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format=RESPONSE_FORMAT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": cleaned_text},
                ],
            )
        except Exception as api_err:
            err_msg = str(api_err)
            if active_backend() == "gemini" and any(k in err_msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    time.sleep(5)
                fallback_model = "gemini-3.5-flash-lite" if model != "gemini-3.5-flash-lite" else "gemini-3.8-flash"
                resp = client.chat.completions.create(
                    model=fallback_model,
                    temperature=0.2,
                    response_format=RESPONSE_FORMAT,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": cleaned_text},
                    ],
                )
            else:
                raise api_err
    except Exception as exc:
        hint = ""
        if active_backend() == "local":
            hint = " (is the local LLM server running at LOCAL_LLM_BASE_URL?)"
        return {"intent": "error", "error": f"[{active_backend()}] request failed: {exc}{hint}"}

    choice = resp.choices[0]
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
    if intent not in ("task", "content", "summary", "query"):
        return {"intent": "error", "error": f"Unknown kind: {intent!r}", "raw": parsed}

    return {"intent": intent, "data": result}

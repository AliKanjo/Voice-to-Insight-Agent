"""
Voice-to-Insight Agent — FastAPI backend.

Pipeline: audio -> /api/transcribe (Whisper STT) -> /api/understand (Gemini/OpenAI/local) -> action.
Includes:
- SQLite persistent database with FTS5 search
- Arabic dialect normalization
- Voice RAG query answering
- Multilingual support (Arabic, English, French, Spanish, etc.)
- Deep Notion integration (priority, due date, direct URL)
"""

import logging
import os
import tempfile
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Never override: variables set by the hosting platform must win over a stray .env file.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import actions
import db
import errors
import rag
import security
import stt
import understand

app = FastAPI(title="Voice-to-Insight Agent")
app.middleware("http")(security.security_middleware)
security.warn_if_open()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Uploads: browsers label recordings audio/* (or video/webm|mp4|ogg for MediaRecorder output).
ALLOWED_AUDIO_EXTS = {".wav", ".webm", ".mp3", ".mp4", ".m4a", ".ogg", ".oga", ".flac", ".mpeg", ".mpga"}
ALLOWED_VIDEO_TYPES = {"video/webm", "video/mp4", "video/ogg"}


def _max_upload_bytes() -> int:
    try:
        return int(float(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024)
    except ValueError:
        return 25 * 1024 * 1024


def _key_configured(name: str) -> bool:
    """True only for real-looking keys, not empty values or .env.example placeholders."""
    value = (os.getenv(name) or "").strip()
    return bool(value) and not value.lower().startswith(("your-", "your_", "changeme", "<"))


@app.middleware("http")
async def reject_oversized_uploads(request, call_next):
    """Refuse oversized uploads from their Content-Length before the body is read.
    (Chunked uploads without a length are still capped inside /api/transcribe.)"""
    if request.url.path == "/api/transcribe":
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = 0
        # Allow some slack for multipart framing around the file itself.
        if length > _max_upload_bytes() + 64 * 1024:
            return JSONResponse(
                {"error": f"Audio file too large (max {_max_upload_bytes() // (1024 * 1024)} MB).",
                 "stage": "transcribe"},
                status_code=413,
            )
    return await call_next(request)


# --- Models ---------------------------------------------------------------
class UnderstandRequest(BaseModel):
    transcript: str
    locale: Optional[str] = None


class RagRequest(BaseModel):
    question: str


# --- Health / config ------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict:
    """Unauthenticated liveness probe for the hosting platform."""
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    """Report backend configuration and active models."""
    return {
        "ok": True,
        "openai_configured": _key_configured("OPENAI_API_KEY"),
        "gemini_configured": _key_configured("GEMINI_API_KEY"),
        "transcribe_backend": stt.active_backend(),
        "understand_backend": understand.active_backend(),
        "notion_configured": _key_configured("NOTION_API_KEY") and bool(os.getenv("NOTION_DATABASE_ID")),
        "database": "sqlite",
        "auth_enabled": security.auth_enabled(),
        "arabic_normalization": True,
        "rag_enabled": True,
    }


# --- Stage 2: speech-to-text ---------------------------------------------
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile) -> JSONResponse:
    """Accept an audio file (WAV from the browser), return transcript + locale."""
    ext = os.path.splitext(audio.filename or "")[1].lower()
    ctype = (audio.content_type or "").split(";")[0].strip().lower()
    if not (ctype.startswith("audio/") or ctype in ALLOWED_VIDEO_TYPES or ext in ALLOWED_AUDIO_EXTS):
        return JSONResponse(
            {"error": f"Unsupported file type ({ctype or ext or 'unknown'}). Upload an audio file.",
             "stage": "transcribe"},
            status_code=415,
        )
    suffix = ext if ext in ALLOWED_AUDIO_EXTS else ".wav"

    max_bytes = _max_upload_bytes()
    tmp_path = None
    try:
        # Copy in chunks so an oversized upload is rejected without loading it into memory.
        size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    return JSONResponse(
                        {"error": f"Audio file too large (max {max_bytes // (1024 * 1024)} MB).",
                         "stage": "transcribe"},
                        status_code=413,
                    )
                tmp.write(chunk)
        if size == 0:
            return JSONResponse({"error": "Empty audio file.", "stage": "transcribe"}, status_code=400)

        # Whisper is CPU-bound and synchronous; keep it off the event loop.
        result = await run_in_threadpool(stt.transcribe, tmp_path)
        return JSONResponse(result)
    except stt.NoSpeechError as exc:
        return JSONResponse({"error": str(exc), "stage": "transcribe"}, status_code=422)
    except stt.STTError as exc:
        msg = errors.public_message(f"Transcription failed: {exc}", "Transcription service failed.", exc=exc)
        return JSONResponse({"error": msg, "stage": "transcribe"}, status_code=502)
    except Exception as exc:
        msg = errors.public_message(f"Unexpected transcription error: {exc}", "Internal server error.", exc=exc)
        return JSONResponse({"error": msg, "stage": "transcribe"}, status_code=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- Stage 3 + 4: understand, then act -----------------------------------
@app.post("/api/understand")
def understand_endpoint(req: UnderstandRequest) -> JSONResponse:
    """Classify + extract, then dispatch the matching action into SQLite / Notion / RAG."""
    result = understand.understand(req.transcript)

    if result.get("intent") == "error":
        if result.get("error_type") == "bad_input":
            return JSONResponse(
                {"stage": "understand", "intent": "error", "error": result["error"]}, status_code=400
            )
        detail = result["error"]
        if "raw" in result:
            detail += f" | raw: {result['raw']!r}"[:2000]
        body = {
            "stage": "understand",
            "intent": "error",
            "error": errors.public_message(detail, "The language model request failed. Please try again."),
        }
        if errors.expose_details() and "raw" in result:
            body["raw"] = result["raw"]
        return JSONResponse(body, status_code=502)

    action = actions.dispatch(
        intent=result["intent"],
        data=result["data"],
        transcript=req.transcript,
        locale=req.locale or "",
    )

    return JSONResponse(
        {
            "intent": result["intent"],
            "data": result["data"],
            "action": action,
        }
    )


# --- Database & Task Management ------------------------------------------
@app.get("/api/tasks")
def get_tasks(status: Optional[str] = None) -> dict:
    """Read tasks from SQLite, newest first."""
    return {"tasks": db.get_tasks(status=status)}


@app.post("/api/tasks/{task_id}/toggle")
def toggle_task(task_id: int) -> JSONResponse:
    """Toggle task status between 'pending' and 'completed'."""
    updated = db.toggle_task(task_id)
    if not updated:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"ok": True, "task": updated})


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: int) -> JSONResponse:
    """Delete a task by ID."""
    if not db.delete_task(task_id):
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.post("/api/tasks/{task_id}/restore")
def restore_task(task_id: int) -> JSONResponse:
    """Undo a delete (available for a few minutes after deleting)."""
    task = db.restore_task(task_id)
    if not task:
        return JSONResponse({"error": "Nothing to restore — the undo window has expired."}, status_code=404)
    return JSONResponse({"ok": True, "task": task})


@app.post("/api/tasks/clear-completed")
def clear_completed() -> JSONResponse:
    """Clear all completed tasks from SQLite."""
    count = db.clear_completed_tasks()
    return JSONResponse({"ok": True, "cleared_count": count})


# --- History & RAG Knowledge Retrieval -----------------------------------
@app.get("/api/history")
def get_history(limit: int = Query(50, ge=1, le=200)) -> dict:
    """Return historical voice notes and insights stored in SQLite."""
    return {"history": db.get_history(limit=limit)}


@app.post("/api/rag/ask")
def rag_ask(req: RagRequest) -> JSONResponse:
    """Search past voice notes and synthesize a grounded answer."""
    if not req.question.strip():
        return JSONResponse({"error": "Empty question."}, status_code=400)
    res = rag.answer_query(req.question)
    return JSONResponse(res, status_code=200 if res.get("ok") else 502)


# --- Notion Status --------------------------------------------------------
@app.get("/api/notion/status")
def notion_status() -> dict:
    api_key = os.getenv("NOTION_API_KEY")
    db_id = os.getenv("NOTION_DATABASE_ID")
    return {
        "configured": bool(api_key and db_id),
        "database_id": db_id if (api_key and db_id) else None,
    }


# --- Frontend -------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

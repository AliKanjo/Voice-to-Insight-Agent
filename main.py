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

import os
import tempfile
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(override=True)

import actions
import db
import rag
import stt
import understand

app = FastAPI(title="Voice-to-Insight Agent")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# --- Models ---------------------------------------------------------------
class UnderstandRequest(BaseModel):
    transcript: str
    locale: Optional[str] = None


class RagRequest(BaseModel):
    question: str


# --- Health / config ------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Report backend configuration and active models."""
    return {
        "ok": True,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "transcribe_backend": stt.active_backend(),
        "understand_backend": understand.active_backend(),
        "notion_configured": bool(os.getenv("NOTION_API_KEY") and os.getenv("NOTION_DATABASE_ID")),
        "database": "sqlite (data/insights.db)",
        "arabic_normalization": True,
        "rag_enabled": True,
    }


# --- Stage 2: speech-to-text ---------------------------------------------
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile) -> JSONResponse:
    """Accept an audio file (WAV from the browser), return transcript + locale."""
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    tmp_path = None
    try:
        data = await audio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        result = stt.transcribe(tmp_path)
        return JSONResponse(result)
    except stt.STTError as exc:
        return JSONResponse({"error": str(exc), "stage": "transcribe"}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": f"Unexpected: {exc}", "stage": "transcribe"}, status_code=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- Stage 3 + 4: understand, then act -----------------------------------
@app.post("/api/understand")
def understand_endpoint(req: UnderstandRequest) -> JSONResponse:
    """Classify + extract, then dispatch the matching action into SQLite / Notion / RAG."""
    result = understand.understand(req.transcript)

    if result.get("intent") == "error":
        return JSONResponse({"stage": "understand", **result})

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
    ok = db.delete_task(task_id)
    return JSONResponse({"ok": ok})


@app.post("/api/tasks/clear-completed")
def clear_completed() -> JSONResponse:
    """Clear all completed tasks from SQLite."""
    count = db.clear_completed_tasks()
    return JSONResponse({"ok": True, "cleared_count": count})


# --- History & RAG Knowledge Retrieval -----------------------------------
@app.get("/api/history")
def get_history(limit: int = 50) -> dict:
    """Return historical voice notes and insights stored in SQLite."""
    return {"history": db.get_history(limit=limit)}


@app.post("/api/rag/ask")
def rag_ask(req: RagRequest) -> JSONResponse:
    """Search past voice notes and synthesize a grounded answer."""
    res = rag.answer_query(req.question)
    return JSONResponse(res)


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

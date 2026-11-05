"""
Voice-to-Insight Agent — FastAPI backend.

Pipeline:  audio -> /api/transcribe (OpenAI STT) -> /api/understand (OpenAI
structured output + action) -> dashboard. The two steps are separate endpoints
so the frontend can render each stage as it completes for a live audience.
"""

import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

import actions
import stt
import understand

app = FastAPI(title="Voice-to-Insight Agent")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# --- Models ---------------------------------------------------------------
class UnderstandRequest(BaseModel):
    transcript: str


# --- Health / config ------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Report which integrations are configured — handy to check before a demo."""
    return {
        "ok": True,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "transcribe_backend": stt.active_backend(),              # "openai" or "local"
        "understand_backend": understand.active_backend(),       # "openai" or "local"
        "notion_configured": bool(os.getenv("NOTION_API_KEY") and os.getenv("NOTION_DATABASE_ID")),
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
    """Classify + extract, then dispatch the matching action. One call, both stages."""
    result = understand.understand(req.transcript)

    if result.get("intent") == "error":
        # 200 so the frontend can render the error card cleanly in-line.
        return JSONResponse({"stage": "understand", **result})

    action = actions.dispatch(result["intent"], result["data"])
    return JSONResponse(
        {
            "intent": result["intent"],
            "data": result["data"],
            "action": action,
        }
    )


# --- Read the running task list ------------------------------------------
@app.get("/api/tasks")
def get_tasks() -> dict:
    return {"tasks": list(reversed(actions.TASKS))}  # newest first


# --- Frontend -------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

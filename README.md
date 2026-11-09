# 🎙️ Voice-to-Insight Agent

Record or upload a voice note in **Arabic (including dialects), English, French, or a mix** → the app transcribes it, classifies the intent, extracts structured data, takes an action, and stores everything so you can ask questions about past notes later.

```
audio ─▶ /api/transcribe ─▶ /api/understand ─▶ action ─▶ SQLite (+ optional Notion)
          (local Whisper        (Gemini, OpenAI,
           or OpenAI)            or local SLM)
```

Each stage runs **either in the cloud or fully on your machine**, flipped independently by one env var.

Intents & extracted data:

| Intent    | Extracted fields                                   | Action                                                  |
|-----------|----------------------------------------------------|---------------------------------------------------------|
| `task`    | `title`, `owner?`, `due_date?`, `due_date_iso?`, `priority` | Saved to SQLite (+ pushed to Notion if configured)      |
| `content` | `platform` (instagram/linkedin), `caption`, `tone` | Saved to SQLite, returns a ready-to-post caption        |
| `summary` | `summary_text`, `key_points[]`                     | Saved to SQLite, returns the structured summary         |
| `query`   | `question`                                         | Voice RAG: searches past notes and answers from them    |

Everything is persisted in SQLite (default `data/insights.db`, configurable with `DB_PATH`) with an FTS5 full-text index used for RAG.

## Files

| File                | Role                                                     |
|---------------------|----------------------------------------------------------|
| `main.py`           | FastAPI app + routes                                     |
| `security.py`       | Optional Basic auth + per-IP rate limiting               |
| `errors.py`         | Error policy: log details, show clients generic messages |
| `stt.py`            | Multilingual transcription (local Whisper or OpenAI)     |
| `understand.py`     | Classification/extraction (Gemini, OpenAI, or local)     |
| `actions.py`        | Intent → action (SQLite storage, Notion sync, RAG)       |
| `db.py`             | SQLite persistence + FTS5 full-text search               |
| `arabic_norm.py`    | Dialect normalization, diacritics & search cleanup       |
| `rag.py`            | Voice RAG over recorded notes                            |
| `static/index.html` | Dashboard UI with tasks, history & RAG cards             |
| `static/app.js`     | Mic capture, upload, task checklist, tabs                |

## Setup

```bash
# 1. Virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Env vars
cp .env.example .env
#   then edit .env and fill in your keys
```

Keys in `.env` — you only need the ones for the backends you pick:

- `GEMINI_API_KEY` — for `UNDERSTAND_BACKEND=gemini` (the `.env.example` default; free key at <https://aistudio.google.com/apikey>)
- `OPENAI_API_KEY` — for `TRANSCRIBE_BACKEND=openai` and/or `UNDERSTAND_BACKEND=openai`
- `NOTION_API_KEY`, `NOTION_DATABASE_ID` — **optional**; if set, `task` items are also pushed to Notion

With `TRANSCRIBE_BACKEND=local` and `UNDERSTAND_BACKEND=local` no keys are needed at all.

If a backend variable is not set at all, the code falls back to `openai` for both stages.

## Run

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000>.

Check integration status any time at <http://localhost:8000/api/health>.

## Deploying

Set these in your hosting platform's environment (they take precedence over any `.env` file):

- `APP_PASSWORD` (and optionally `APP_USERNAME`, default `admin`) — turns on HTTP Basic auth for the UI and API. **Without it the app is open to anyone.**
- `DB_PATH` — put the SQLite file on a persistent volume, e.g. `/var/data/insights.db`.
- `RATE_LIMIT_PER_MINUTE` — per-IP limit on the transcribe / understand / RAG endpoints (default 20).
- `APP_TIMEZONE` — your time zone (e.g. `Asia/Beirut`), so "tomorrow" resolves to the right date.
- `LLM_TIMEOUT_SECONDS` / `STT_TIMEOUT_SECONDS` — give up on a slow provider after this long (defaults 60 / 120).

Run without `--reload`, and pass proxy headers through if you sit behind a load balancer:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
```

`/healthz` is an unauthenticated liveness probe for the platform's health checks.

## Demo tips (live path)

- **Start green without a mic:** you only need the key for your understanding backend. Paste text into the fallback box (or click a **Sample** button) and hit **Run text → understand**. The whole classify → extract → act chain runs without touching the mic or transcription at all.
- Then wire the mic: click **Record**, speak, click **Stop** (or **Upload Audio**, max `MAX_UPLOAD_MB`, default 25 MB). The recorded audio is sent **as-is** to the transcription backend — no in-browser conversion.
- Every pipeline stage renders as it completes (transcript → classification → action) so the audience watches the chain, not just the final result.
- **Errors are loud (when you want them to be):** failures show in a red banner at the top of the page. Set `EXPOSE_ERROR_DETAILS=true` locally to see the real provider message there; leave it off in production, where clients get a generic message and the details go to the server log.

### Transcription backend: cloud vs. local Whisper

| `TRANSCRIBE_BACKEND` | Ears                                        | Needs                     |
|----------------------|---------------------------------------------|---------------------------|
| `openai` (default)   | `gpt-4o-transcribe` via OpenAI API          | `OPENAI_API_KEY`          |
| `local`              | OpenAI's **Whisper** weights via `faster-whisper` | `pip install faster-whisper` |

`faster-whisper` runs the *same* Whisper weights as OpenAI's reference implementation, just on the CTranslate2 runtime — same accuracy, ~4x faster, no `torch` or `ffmpeg` binary required.

```bash
# in .env:
TRANSCRIBE_BACKEND=local
WHISPER_MODEL=small      # tiny | base | small | medium | large-v3
```

The model is downloaded on first use and cached; it stays loaded in memory between requests, so only the **first** transcription pays the load cost. Warm it up once before going live.

On quality: `small` is fine for English, **`medium` is the realistic floor for Lebanese-dialect Arabic**. Unlike the OpenAI path, local Whisper returns a real detected language (shown in the UI with its confidence, e.g. `ar (98%)`) — but it commits to *one* language per clip, so heavily code-switched audio gets labelled by whichever dominates.

### Understanding backend: cloud vs. local SLM

The understanding layer has three interchangeable brains behind the **same strict-JSON contract**, selected by `UNDERSTAND_BACKEND`:

| `UNDERSTAND_BACKEND` | Brain                                  | Needs                          |
|----------------------|----------------------------------------|--------------------------------|
| `gemini`             | Gemini Flash-Lite (`GEMINI_MODEL`)     | `GEMINI_API_KEY`               |
| `openai` (default)   | `gpt-4o` via OpenAI API                | `OPENAI_API_KEY`               |
| `local`              | Qwen2.5-1.5B-Instruct via **llama.cpp**| a running `llama-server`       |

Because llama.cpp constrains generation to the JSON schema (via grammar), even a 1.5B model **cannot** emit invalid JSON — it's forced into shape token by token. The tradeoff is *semantic* quality: a 1.5B model is solid on English intents and lighter on nuanced or Lebanese-dialect Arabic. Keep `gemini` or `openai` as your live fallback — flip one env var if the local model wobbles on stage.

> Set `TRANSCRIBE_BACKEND=local` **and** `UNDERSTAND_BACKEND=local` for a fully offline pipeline — no network calls at all.

**Run the local model:**

```bash
# 1. Install llama.cpp (macOS). Use a recent build — json_schema response_format
#    support landed in mid-2024.
brew install llama.cpp

# 2. Serve Qwen2.5-1.5B-Instruct (auto-downloads the GGUF on first run)
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF -c 4096
#    -> serves an OpenAI-compatible API at http://localhost:8080

# 3. Point the app at it
#    in .env:
#      UNDERSTAND_BACKEND=local
#      LOCAL_LLM_BASE_URL=http://localhost:8080/v1
```

Confirm which brain is live at <http://localhost:8000/api/health> (`understand_backend` field). Want the smallest possible model for the "tiny SLM" story? Swap the `-hf` flag to `Qwen/Qwen2.5-0.5B-Instruct-GGUF` and set `LOCAL_LLM_MODEL=qwen2.5-0.5b-instruct` — expect weaker Arabic.

### Language detection

With `TRANSCRIBE_BACKEND=openai`, `gpt-4o-transcribe` auto-detects language and handles code-switched Arabic/English in one pass; the locale badge (e.g. `Arabic`, `Mixed (Arabic + English)`) is derived from the scripts present in the transcript. Prefer an explicit language field? Switch `OPENAI_STT_MODEL` in `stt.py` to `whisper-1` and request `verbose_json`. Local Whisper reports its detected language and confidence directly.

### Notion (optional)

The push in `actions.py` assumes your Notion database has a title property named **`Name`**, and optionally a **Select** property for priority (Low / Medium / High) and a **Date** property for the due date. Their names default to `Priority` and `Due Date`; change them with `NOTION_PRIORITY_PROPERTY` and `NOTION_DUE_PROPERTY`. If Notion rejects the payload because a column is missing, the task is retried without the due date, then with the title only.

### Due dates

For tasks, the model returns both the deadline as spoken (`due_date`, e.g. "before Friday") and the resolved calendar date (`due_date_iso`, e.g. `2026-11-13`). Relative dates are resolved against today's date in `APP_TIMEZONE` (default UTC). Invalid dates from the model are dropped. The dashboard shows the resolved date and flags pending tasks past it as **overdue**.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite uses a fake language model and a temporary SQLite database, so it needs no API keys or network. GitHub Actions runs it on every pull request and push to `main` (`.github/workflows/ci.yml`), along with an import check for `faster-whisper` and a syntax check of `static/app.js`.

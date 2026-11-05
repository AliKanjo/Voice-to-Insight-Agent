# 🎙️ Voice-to-Insight Agent

A small demo app for a live coding webinar. Record a voice note in **Arabic, English, or a mix of both** → the app transcribes it, classifies the intent, extracts structured data, and takes an action.

```
audio ─▶ /api/transcribe ─▶ /api/understand ─▶ action
          (OpenAI or           (OpenAI or
           local Whisper)       local SLM)
```

Each stage runs **either in the cloud or fully on your machine**, flipped independently by one env var — so you can demo a 100% offline pipeline and still fall back mid-stream.

Intents & extracted data:

| Intent    | Extracted fields                                  | Action                                   |
|-----------|---------------------------------------------------|------------------------------------------|
| `task`    | `title`, `owner?`, `due_date?`, `priority`        | Add to in-memory list (+ optional Notion) |
| `content` | `platform` (instagram/linkedin), `caption`, `tone`| Return a ready-to-post caption           |
| `summary` | `summary_text`, `key_points[]`                    | Return the structured summary            |

Persistent SQLite database (data/insights.db) with FTS5 search.

## Files

| File                | Role                                              |
|---------------------|---------------------------------------------------|
| main.py           | FastAPI app + routes                              |
| stt.py            | Multilingual transcription (Whisper or OpenAI)    |
| understand.py     | Classification/extraction (Gemini, OpenAI, local) |
| ctions.py        | Intent -> action (SQLite storage, Notion sync, RAG)|
| db.py             | SQLite persistence + FTS5 full-text search        |
| rabic_norm.py    | Dialect normalization, diacritics & search clean  |
| 
ag.py            | Voice RAG search engine over recorded notes       |
| static/index.html | Dashboard UI with tasks, history & RAG cards      |
| static/app.js     | Mic capture, interactive task checklist, tabs     |

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

Required keys in `.env`:

- `OPENAI_API_KEY` — used for **both** transcription and the understanding layer (the only key you need)
- `NOTION_API_KEY`, `NOTION_DATABASE_ID` — **optional**; if set, `task` items are also pushed to Notion

## Run

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000>.

Check integration status any time at <http://localhost:8000/api/health>.

## Demo tips (live path)

- **Start green without a mic:** you only need `OPENAI_API_KEY`. Paste text into the fallback box (or click a **Sample** button) and hit **Run text → understand**. The whole classify → extract → act chain runs without touching the mic or transcription at all.
- Then wire the mic: click **Record**, speak, click **Stop**. The recorded audio is sent **as-is** to OpenAI's transcription API — no `ffmpeg` and no in-browser conversion.
- Every pipeline stage renders as it completes (transcript → classification → action) so the audience watches the chain, not just the final result.
- **Errors are loud:** any OpenAI/Notion failure shows in a red banner at the top of the page with the actual message, so you can debug on stage without guessing.

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

The understanding layer has two interchangeable brains behind the **same strict-JSON contract**, selected by `UNDERSTAND_BACKEND`:

| `UNDERSTAND_BACKEND` | Brain                                  | Needs                          |
|----------------------|----------------------------------------|--------------------------------|
| `openai` (default)   | `gpt-4o` via OpenAI API                | `OPENAI_API_KEY`               |
| `local`              | Qwen2.5-1.5B-Instruct via **llama.cpp**| a running `llama-server`       |

Because llama.cpp constrains generation to the JSON schema (via grammar), even a 1.5B model **cannot** emit invalid JSON — it's forced into shape token by token. The tradeoff is *semantic* quality: a 1.5B model is solid on English intents and lighter on nuanced or Lebanese-dialect Arabic. Keep `openai` as your live fallback — flip one env var if the local model wobbles on stage.

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

Transcription uses `gpt-4o-transcribe`, which auto-detects language and handles code-switched Arabic/English in one pass. The locale badge (`ar` / `en` / `mixed`) is derived from the script present in the transcript. Prefer an explicit language field? Switch `MODEL` in `stt.py` to `whisper-1` and request `verbose_json`.

### Notion (optional)

The push in `actions.py` assumes your Notion database has a title property named **`Name`**. Adjust the `properties` payload there if yours differs.

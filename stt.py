"""
Speech-to-text with two interchangeable backends, chosen at startup via the
TRANSCRIBE_BACKEND env var:

  - "openai" (default): gpt-4o-transcribe via the OpenAI API. Nothing to install.
  - "local":            OpenAI's Whisper weights running on your machine via
                        faster-whisper (CTranslate2 runtime). Fully offline.

Both return {"transcript": str, "locale": str} and raise STTError on failure,
so main.py doesn't care which one is active.

Both accept the browser's recorded audio (webm/mp4/wav/m4a/mp3) directly —
no ffmpeg binary and no in-browser conversion needed.
"""

import os

from openai import OpenAI

# --- OpenAI API backend ----------------------------------------------------
OPENAI_STT_MODEL = "gpt-4o-transcribe"

# --- Local Whisper backend -------------------------------------------------
# tiny | base | small | medium | large-v3. "small" is a good live-demo balance;
# bump to "medium" if Lebanese-dialect Arabic needs the extra accuracy.
DEFAULT_WHISPER_MODEL = "small"

# Loading Whisper weights takes seconds, so keep one instance alive per config.
_MODEL_CACHE: dict = {}


class STTError(RuntimeError):
    """Raised when transcription fails, so main.py can surface it loudly."""


def active_backend() -> str:
    return os.getenv("TRANSCRIBE_BACKEND", "openai").strip().lower()


def _detect_locale(text: str) -> str:
    """Rough label from the script(s) present — used for the OpenAI API path,
    which doesn't return a language field."""
    has_ar = any("؀" <= c <= "ۿ" for c in text)
    has_latin = any(("a" <= c.lower() <= "z") for c in text)
    if has_ar and has_latin:
        return "mixed (ar + en)"
    if has_ar:
        return "ar"
    if has_latin:
        return "en"
    return "unknown"


# --- Backend: OpenAI API ---------------------------------------------------
def _transcribe_openai(audio_path: str) -> dict:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise STTError("OPENAI_API_KEY is not set — add it to your .env file.")

    try:
        client = OpenAI(api_key=key)
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=OPENAI_STT_MODEL,
                file=f,
                # No `language=` → let the model auto-detect (needed for code-switching).
                prompt="The audio may mix Arabic (including Lebanese dialect) and English.",
            )
    except Exception as exc:
        raise STTError(f"OpenAI transcription failed: {exc}") from exc

    text = (getattr(result, "text", "") or "").strip()
    return {"transcript": text, "locale": _detect_locale(text)}


# --- Backend: local Whisper via faster-whisper ------------------------------
def _get_local_model():
    """Lazily load + cache the Whisper model. Imported here so the faster-whisper
    dependency is only required when you actually use the local backend."""
    size = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    device = os.getenv("WHISPER_DEVICE", "auto")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    cache_key = (size, device, compute_type)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    try:
        # First run downloads the weights from Hugging Face, then they're cached.
        model = WhisperModel(size, device=device, compute_type=compute_type)
    except Exception as exc:
        raise STTError(f"Could not load local Whisper model '{size}': {exc}") from exc

    _MODEL_CACHE[cache_key] = model
    return model


def _transcribe_local(audio_path: str) -> dict:
    model = _get_local_model()
    try:
        # No `language=` → Whisper auto-detects. It commits to ONE language per
        # clip, so heavily code-switched audio gets labelled by the dominant one.
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
    except Exception as exc:
        raise STTError(f"Local Whisper transcription failed: {exc}") from exc

    locale = info.language or "unknown"
    if getattr(info, "language_probability", None):
        locale = f"{locale} ({info.language_probability:.0%})"
    return {"transcript": text, "locale": locale}


# --- Entry point -----------------------------------------------------------
def transcribe(audio_path: str) -> dict:
    """Return {"transcript": str, "locale": str}. Raises STTError on failure."""
    backend = active_backend()
    result = _transcribe_local(audio_path) if backend == "local" else _transcribe_openai(audio_path)

    if not result["transcript"]:
        raise STTError("Transcription returned empty text — was there any speech?")
    return result

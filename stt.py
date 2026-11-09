"""
Speech-to-text with interchangeable backends ("local" Whisper or "openai" API).
Supports multilingual audio (Arabic dialects, English, French, Spanish, German, Turkish, etc.).
"""

import os

# PyAV 14+ / 19+ removed the legacy `metadata_errors` keyword argument, but
# faster-whisper 1.2.1 still passes `metadata_errors="ignore"` to av.open().
# This patch intercepts av.open() to strip `metadata_errors` safely.
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except Exception:
    pass

from openai import OpenAI

import arabic_norm

# --- OpenAI API backend ----------------------------------------------------
OPENAI_STT_MODEL = "gpt-4o-transcribe"

# --- Local Whisper backend -------------------------------------------------
DEFAULT_WHISPER_MODEL = "small"

_MODEL_CACHE: dict = {}

_LANG_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "tr": "Turkish",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ur": "Urdu",
    "hi": "Hindi",
    "fa": "Persian",
}


class STTError(RuntimeError):
    """Raised when transcription fails, so main.py can surface it loudly."""


def active_backend() -> str:
    return os.getenv("TRANSCRIBE_BACKEND", "openai").strip().lower()


def _format_locale(code: str, prob: float = None) -> str:
    name = _LANG_NAMES.get(code.lower(), code.upper())
    if prob is not None:
        return f"{name} ({prob:.0%})"
    return name


def _detect_locale(text: str) -> str:
    """Rough label from scripts present when model doesn't output language."""
    has_ar = any("؀" <= c <= "ۿ" for c in text)
    has_latin = any(("a" <= c.lower() <= "z") for c in text)
    has_cyrillic = any("\u0400" <= c <= "\u04FF" for c in text)
    if has_ar and has_latin:
        return "Mixed (Arabic + English)"
    if has_ar:
        return "Arabic"
    if has_cyrillic:
        return "Russian"
    if has_latin:
        return "English / European"
    return "Unknown"


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
                prompt="The audio may mix Arabic (including dialects), English, French, or other languages.",
            )
    except Exception as exc:
        raise STTError(f"OpenAI transcription failed: {exc}") from exc

    raw_text = (getattr(result, "text", "") or "").strip()
    clean_text = arabic_norm.clean_transcript(raw_text)
    return {"transcript": clean_text, "locale": _detect_locale(clean_text)}


# --- Backend: local Whisper via faster-whisper ------------------------------
def _get_local_model():
    size = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    cache_key = (size, device, compute_type)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTError(
            f"Could not import faster-whisper ({exc}). "
            "Install the pinned dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        model = WhisperModel(size, device=device, compute_type=compute_type)
    except Exception as exc:
        raise STTError(f"Could not load local Whisper model '{size}': {exc}") from exc

    _MODEL_CACHE[cache_key] = model
    return model


def _transcribe_local(audio_path: str) -> dict:
    model = _get_local_model()
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        raw_text = "".join(seg.text for seg in segments).strip()
    except Exception as exc:
        raise STTError(f"Local Whisper transcription failed: {exc}") from exc

    clean_text = arabic_norm.clean_transcript(raw_text)
    lang_code = info.language or "unknown"
    prob = getattr(info, "language_probability", None)
    locale_label = _format_locale(lang_code, prob)

    return {"transcript": clean_text, "locale": locale_label}


def transcribe(audio_path: str) -> dict:
    backend = active_backend()
    result = _transcribe_local(audio_path) if backend == "local" else _transcribe_openai(audio_path)

    if not result["transcript"]:
        raise STTError("Transcription returned empty text — was there any speech?")
    return result

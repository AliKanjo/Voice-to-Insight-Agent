"""
Arabic normalization module.

Handles dialect variations, orthographic inconsistencies, diacritics, and numeral conversions
commonly produced by speech-to-text systems (e.g., Whisper).
"""

import re

# Arabic diacritics (Tashkeel) Unicode range
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")

# Tatweel / Kashida
_TATWEEL = re.compile(r"\u0640")

# Eastern Arabic numerals to standard Western digits
_EASTERN_TO_WESTERN_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Persian/Urdu numerals to standard Western digits
_PERSIAN_TO_WESTERN_NUMS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Alef variations -> standard bare Alef
_ALEF_MAP = str.maketrans("أإآٱ", "اااا")


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (Fatha, Damma, Kasra, Tanween, Sukun, Shadda, Dagger Alef)."""
    if not text:
        return ""
    return _DIACRITICS.sub("", text)


def strip_tatweel(text: str) -> str:
    """Remove Arabic elongation mark (ـ)."""
    if not text:
        return ""
    return _TATWEEL.sub("", text)


def normalize_numerals(text: str) -> str:
    """Convert Eastern Arabic (٠-٩) and Persian (۰-۹) numerals to Western digits (0-9)."""
    if not text:
        return ""
    return text.translate(_EASTERN_TO_WESTERN_NUMS).translate(_PERSIAN_TO_WESTERN_NUMS)


def normalize_alef(text: str) -> str:
    """Normalize all Alef forms (أ, إ, آ, ٱ) to a bare Alef (ا)."""
    if not text:
        return ""
    return text.translate(_ALEF_MAP)


def normalize_yaa(text: str) -> str:
    """Normalize Alef Maqsura (ى) to standard Yaa (ي)."""
    if not text:
        return ""
    return text.replace("ى", "ي")


def normalize_taa_marbuta(text: str) -> str:
    """Normalize Taa Marbuta (ة) to Haa (ه) for search indexing."""
    if not text:
        return ""
    return text.replace("ة", "ه")


def normalize_for_search(text: str) -> str:
    """
    Full aggressive normalization suitable for search indexing (FTS5 / search queries).
    Ensures that 'أحمد' matches 'احمد', 'مدرسة' matches 'مدرسه', 'إلى' matches 'الي'.
    """
    if not text:
        return ""
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = normalize_numerals(text)
    text = normalize_alef(text)
    text = normalize_yaa(text)
    text = normalize_taa_marbuta(text)
    # Collapse extra whitespace
    return " ".join(text.split()).strip().lower()


def clean_transcript(text: str) -> str:
    """
    Standard cleanup for spoken transcripts while preserving readability:
    - Strips tashkeel and tatweel
    - Converts eastern numerals to digits
    - Preserves letters (keeps أ, إ, ة for natural display)
    - Normalizes punctuation and excessive spaces
    """
    if not text:
        return ""
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = normalize_numerals(text)
    # Clean redundant spaces before punctuation
    text = re.sub(r"\s+([.,!؟?؛:])", r"\1", text)
    return " ".join(text.split()).strip()


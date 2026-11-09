"""
Error reporting policy.

Internal details (provider error bodies, exception text, file paths) are always
logged on the server. Clients only see them when EXPOSE_ERROR_DETAILS is true,
which is handy for local demos; in production they get a generic message.
"""

import logging
import os

log = logging.getLogger("voice_to_insight")


def expose_details() -> bool:
    return os.getenv("EXPOSE_ERROR_DETAILS", "").strip().lower() in ("1", "true", "yes", "on")


def public_message(detail: str, generic: str, *, exc: BaseException | None = None) -> str:
    """Log `detail` and return what the client may see."""
    log.error("%s", detail, exc_info=exc)
    return detail if expose_details() else generic

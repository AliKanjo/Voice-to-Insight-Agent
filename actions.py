"""
Action layer: branch on the classified intent and *do* something with it.

- task    -> append to an in-memory list (+ optional Notion push)
- content -> format the caption for display
- summary -> return the structured summary

No database. TASKS is a module-level list that lives for the process lifetime,
which is all a demo needs.
"""

import os
from datetime import datetime, timezone

import httpx

# In-memory "database". Newest first is handled at read time in main.py.
TASKS: list[dict] = []

_NOTION_VERSION = "2022-06-28"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _push_to_notion(task: dict) -> dict:
    """Best-effort push of a task to a Notion database. Returns a status dict."""
    api_key = os.getenv("NOTION_API_KEY")
    db_id = os.getenv("NOTION_DATABASE_ID")
    if not api_key or not db_id:
        return {"pushed": False, "reason": "Notion not configured (optional)."}

    # Assumes the DB has a title property named "Name". Extra props are best-effort.
    payload = {
        "parent": {"database_id": db_id},
        "properties": {
            "Name": {"title": [{"text": {"content": task["title"]}}]},
        },
    }
    try:
        resp = httpx.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": _NOTION_VERSION,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15.0,
        )
        if resp.status_code >= 400:
            return {"pushed": False, "reason": f"Notion {resp.status_code}: {resp.text[:300]}"}
        return {"pushed": True, "page_id": resp.json().get("id")}
    except Exception as exc:
        return {"pushed": False, "reason": f"Notion request failed: {exc}"}


def dispatch(intent: str, data: dict) -> dict:
    """Take the action for a given intent and return a display-ready result."""
    if intent == "task":
        task = {
            "id": len(TASKS) + 1,
            "title": data.get("title", "(untitled task)"),
            "owner": data.get("owner"),
            "due_date": data.get("due_date"),
            "priority": data.get("priority", "medium"),
            "created_at": _now_iso(),
        }
        TASKS.append(task)
        notion = _push_to_notion(task)
        return {"type": "task_added", "task": task, "notion": notion}

    if intent == "content":
        return {
            "type": "caption_generated",
            "platform": data.get("platform"),
            "caption": data.get("caption", ""),
            "tone": data.get("tone"),
        }

    if intent == "summary":
        return {
            "type": "summary_shown",
            "summary_text": data.get("summary_text", ""),
            "key_points": data.get("key_points", []),
        }

    return {"type": "noop", "reason": f"Unknown intent: {intent}"}

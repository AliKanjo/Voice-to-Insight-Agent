"""
Action layer: branch on the classified intent and execute the corresponding action.

- task    -> save to SQLite database + sync to Notion (with Priority, Due Date & Direct URL)
- content -> save to SQLite database + return ready-to-post caption
- summary -> save to SQLite database + return structured summary
- query   -> run RAG search over past voice notes & synthesize grounded answer
"""

import os
from typing import Optional

import httpx

import db
import errors
import rag

_NOTION_VERSION = "2022-06-28"


def _format_notion_url(page_id: str) -> str:
    clean_id = page_id.replace("-", "")
    return f"https://notion.so/{clean_id}"


def _notion_post(api_key: str, payload: dict) -> httpx.Response:
    return httpx.post(
        "https://api.notion.com/v1/pages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=15.0,
    )


def _push_to_notion(task: dict) -> dict:
    """Push a task to Notion with title, priority (select) and due date (date).

    Column names are configurable (NOTION_PRIORITY_PROPERTY, NOTION_DUE_PROPERTY).
    If the database rejects the payload (e.g. a column is missing or has another
    type), retry without the due date, then with the title only."""
    api_key = os.getenv("NOTION_API_KEY")
    db_id = os.getenv("NOTION_DATABASE_ID")
    if not api_key or not db_id:
        return {"pushed": False, "reason": "Notion not configured (optional)."}

    title = task.get("title") or "(untitled task)"
    priority = (task.get("priority") or "medium").capitalize()
    due_iso = task.get("due_date_iso")
    priority_prop = os.getenv("NOTION_PRIORITY_PROPERTY", "Priority")
    due_prop = os.getenv("NOTION_DUE_PROPERTY", "Due Date")

    title_only = {"Name": {"title": [{"text": {"content": title}}]}}
    with_priority = {**title_only, priority_prop: {"select": {"name": priority}}}
    attempts = []
    if due_iso:
        attempts.append({**with_priority, due_prop: {"date": {"start": due_iso}}})
    attempts += [with_priority, title_only]

    try:
        resp = None
        for properties in attempts:
            resp = _notion_post(api_key, {"parent": {"database_id": db_id}, "properties": properties})
            # Only schema/validation errors are worth retrying with fewer properties.
            if resp.status_code != 400:
                break
        if resp.status_code < 400:
            page_id = resp.json().get("id")
            return {"pushed": True, "page_id": page_id, "url": _format_notion_url(page_id)}
        reason = errors.public_message(
            f"Notion {resp.status_code}: {resp.text[:300]}",
            f"Notion sync failed (HTTP {resp.status_code}).",
        )
        return {"pushed": False, "reason": reason}
    except Exception as exc:
        reason = errors.public_message(f"Notion request failed: {exc}", "Notion sync failed.", exc=exc)
        return {"pushed": False, "reason": reason}


def dispatch(
    intent: str,
    data: dict,
    transcript: str = "",
    locale: str = "",
    user_id: Optional[int] = None,
) -> dict:
    """Take the action for a given intent, persist in SQLite, and return display-ready result."""

    # 1. Save voice note in SQLite (for search & history)
    note_id = None
    if transcript and intent != "query":
        try:
            note_id = db.save_voice_note(
                transcript=transcript,
                locale=locale,
                intent=intent,
                raw_data=data,
                user_id=user_id,
            )
        except Exception:
            note_id = None

    # --- Intent: Task ---------------------------------------------------------
    if intent == "task":
        title = data.get("title", "(untitled task)")
        owner = data.get("owner")
        due_date = data.get("due_date")
        due_date_iso = data.get("due_date_iso")
        priority = data.get("priority", "medium")

        # Push to Notion (optional)
        notion = _push_to_notion({
            "title": title,
            "owner": owner,
            "due_date": due_date,
            "due_date_iso": due_date_iso,
            "priority": priority,
        })

        notion_page_id = notion.get("page_id") if notion.get("pushed") else None
        notion_url = notion.get("url") if notion.get("pushed") else None

        # Persist task in SQLite
        saved_task = db.save_task(
            title=title,
            owner=owner,
            due_date=due_date,
            due_date_iso=due_date_iso,
            priority=priority,
            note_id=note_id,
            notion_page_id=notion_page_id,
            notion_url=notion_url,
            user_id=user_id,
        )

        return {
            "type": "task_added",
            "task": saved_task,
            "notion": notion,
        }

    # --- Intent: Content ------------------------------------------------------
    if intent == "content":
        platform = data.get("platform")
        caption = data.get("caption", "")
        tone = data.get("tone")

        saved_content = db.save_content(
            platform=platform,
            caption=caption,
            tone=tone,
            note_id=note_id,
            user_id=user_id,
        )

        return {
            "type": "caption_generated",
            "content_id": saved_content["id"],
            "platform": platform,
            "caption": caption,
            "tone": tone,
        }

    # --- Intent: Summary ------------------------------------------------------
    if intent == "summary":
        summary_text = data.get("summary_text", "")
        key_points = data.get("key_points", [])

        saved_summary = db.save_summary(
            summary_text=summary_text,
            key_points=key_points,
            note_id=note_id,
            user_id=user_id,
        )

        return {
            "type": "summary_shown",
            "summary_id": saved_summary["id"],
            "summary_text": summary_text,
            "key_points": key_points,
        }

    # --- Intent: Query (Voice RAG) -------------------------------------------
    if intent == "query":
        question = data.get("question") or transcript
        rag_result = rag.answer_query(question, user_id=user_id)

        return {
            "type": "rag_answer",
            "question": rag_result["question"],
            "answer": rag_result["answer"],
            "citations": rag_result["citations"],
            "context_count": rag_result["context_count"],
        }

    return {"type": "noop", "reason": f"Unknown intent: {intent}"}

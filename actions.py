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


def _push_to_notion(task: dict) -> dict:
    """Push a task to Notion with title, priority tag, and due date."""
    api_key = os.getenv("NOTION_API_KEY")
    db_id = os.getenv("NOTION_DATABASE_ID")
    if not api_key or not db_id:
        return {"pushed": False, "reason": "Notion not configured (optional)."}

    title = task.get("title", "(untitled task)")
    priority = (task.get("priority") or "medium").capitalize()
    due_date = task.get("due_date")

    # Build properties payload
    properties = {
        "Name": {"title": [{"text": {"content": title}}]},
    }

    # Attempt to include Priority select if user configured a select column named "Priority"
    if priority:
        properties["Priority"] = {"select": {"name": priority}}

    payload = {
        "parent": {"database_id": db_id},
        "properties": properties,
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
            # If failed due to strict property name mismatch (e.g. "Priority" column missing), retry with just title
            fallback_payload = {
                "parent": {"database_id": db_id},
                "properties": {"Name": {"title": [{"text": {"content": title}}]}},
            }
            fb_resp = httpx.post(
                "https://api.notion.com/v1/pages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Notion-Version": _NOTION_VERSION,
                    "Content-Type": "application/json",
                },
                json=fallback_payload,
                timeout=15.0,
            )
            if fb_resp.status_code < 400:
                page_id = fb_resp.json().get("id")
                return {"pushed": True, "page_id": page_id, "url": _format_notion_url(page_id)}
            reason = errors.public_message(
                f"Notion {fb_resp.status_code}: {fb_resp.text[:300]}",
                f"Notion sync failed (HTTP {fb_resp.status_code}).",
            )
            return {"pushed": False, "reason": reason}

        page_id = resp.json().get("id")
        return {"pushed": True, "page_id": page_id, "url": _format_notion_url(page_id)}
    except Exception as exc:
        reason = errors.public_message(f"Notion request failed: {exc}", "Notion sync failed.", exc=exc)
        return {"pushed": False, "reason": reason}


def dispatch(intent: str, data: dict, transcript: str = "", locale: str = "") -> dict:
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
            )
        except Exception:
            note_id = None

    # --- Intent: Task ---------------------------------------------------------
    if intent == "task":
        title = data.get("title", "(untitled task)")
        owner = data.get("owner")
        due_date = data.get("due_date")
        priority = data.get("priority", "medium")

        # Push to Notion (optional)
        notion = _push_to_notion({
            "title": title,
            "owner": owner,
            "due_date": due_date,
            "priority": priority,
        })

        notion_page_id = notion.get("page_id") if notion.get("pushed") else None
        notion_url = notion.get("url") if notion.get("pushed") else None

        # Persist task in SQLite
        saved_task = db.save_task(
            title=title,
            owner=owner,
            due_date=due_date,
            priority=priority,
            note_id=note_id,
            notion_page_id=notion_page_id,
            notion_url=notion_url,
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
        rag_result = rag.answer_query(question)

        return {
            "type": "rag_answer",
            "question": rag_result["question"],
            "answer": rag_result["answer"],
            "citations": rag_result["citations"],
            "context_count": rag_result["context_count"],
        }

    return {"type": "noop", "reason": f"Unknown intent: {intent}"}

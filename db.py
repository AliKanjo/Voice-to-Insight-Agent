"""
Database layer for Voice-to-Insight Agent.

Uses SQLite with FTS5 for local persistence, task status management,
and fast full-text search for RAG knowledge retrieval.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

import arabic_norm

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "insights.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Initialize SQLite tables and FTS5 search index."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS voice_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript TEXT NOT NULL,
                normalized_transcript TEXT NOT NULL,
                locale TEXT,
                intent TEXT NOT NULL,
                raw_data TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER REFERENCES voice_notes(id) ON DELETE SET NULL,
                title TEXT NOT NULL,
                owner TEXT,
                due_date TEXT,
                priority TEXT NOT NULL DEFAULT 'medium',
                status TEXT NOT NULL DEFAULT 'pending',
                notion_page_id TEXT,
                notion_url TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS content_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER REFERENCES voice_notes(id) ON DELETE SET NULL,
                platform TEXT,
                caption TEXT NOT NULL,
                tone TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER REFERENCES voice_notes(id) ON DELETE SET NULL,
                summary_text TEXT NOT NULL,
                key_points TEXT,
                created_at TEXT NOT NULL
            );

            -- FTS5 virtual table for full-text search across transcripts, tasks, and summaries
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                note_id UNINDEXED,
                title,
                content,
                normalized_content,
                tokenize = 'unicode61'
            );
        """)


# --- Voice Note Operations ------------------------------------------------
def save_voice_note(transcript: str, locale: str, intent: str, raw_data: dict) -> int:
    """Insert a recorded voice note and update the search index."""
    norm = arabic_norm.normalize_for_search(transcript)
    created_at = _now_iso()
    raw_json = json.dumps(raw_data, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO voice_notes (transcript, normalized_transcript, locale, intent, raw_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (transcript, norm, locale, intent, raw_json, created_at),
        )
        note_id = cur.lastrowid

        # Index in FTS
        cur.execute(
            """
            INSERT INTO notes_fts (note_id, title, content, normalized_content)
            VALUES (?, ?, ?, ?)
            """,
            (note_id, intent, transcript, norm),
        )
        conn.commit()
        return note_id


# --- Task Operations ------------------------------------------------------
def save_task(
    title: str,
    owner: Optional[str] = None,
    due_date: Optional[str] = None,
    priority: str = "medium",
    note_id: Optional[int] = None,
    notion_page_id: Optional[str] = None,
    notion_url: Optional[str] = None,
) -> dict:
    created_at = _now_iso()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks (note_id, title, owner, due_date, priority, status, notion_page_id, notion_url, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (note_id, title, owner, due_date, priority, notion_page_id, notion_url, created_at),
        )
        task_id = cur.lastrowid
        conn.commit()

        return {
            "id": task_id,
            "note_id": note_id,
            "title": title,
            "owner": owner,
            "due_date": due_date,
            "priority": priority,
            "status": "pending",
            "notion_page_id": notion_page_id,
            "notion_url": notion_url,
            "created_at": created_at,
        }


def get_tasks(status: Optional[str] = None) -> list[dict]:
    with get_connection() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY id DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def toggle_task(task_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return None
        current_status = row["status"]
        new_status = "completed" if current_status == "pending" else "pending"
        completed_at = _now_iso() if new_status == "completed" else None

        conn.execute(
            "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
            (new_status, completed_at, task_id),
        )
        conn.commit()

        updated = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(updated)


def delete_task(task_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        return cur.rowcount > 0


def clear_completed_tasks() -> int:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE status = 'completed'")
        conn.commit()
        return cur.rowcount


# --- Content Item Operations ----------------------------------------------
def save_content(
    platform: Optional[str], caption: str, tone: Optional[str], note_id: Optional[int] = None
) -> dict:
    created_at = _now_iso()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO content_items (note_id, platform, caption, tone, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (note_id, platform, caption, tone, created_at),
        )
        content_id = cur.lastrowid
        conn.commit()
        return {
            "id": content_id,
            "note_id": note_id,
            "platform": platform,
            "caption": caption,
            "tone": tone,
            "created_at": created_at,
        }


# --- Summary Operations ---------------------------------------------------
def save_summary(
    summary_text: str, key_points: list[str], note_id: Optional[int] = None
) -> dict:
    created_at = _now_iso()
    points_json = json.dumps(key_points, ensure_ascii=False)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO summaries (note_id, summary_text, key_points, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (note_id, summary_text, points_json, created_at),
        )
        summary_id = cur.lastrowid
        conn.commit()
        return {
            "id": summary_id,
            "note_id": note_id,
            "summary_text": summary_text,
            "key_points": key_points,
            "created_at": created_at,
        }


# --- History & RAG Retrieval ----------------------------------------------
def get_history(limit: int = 50) -> list[dict]:
    """Retrieve recent voice notes with their classified data."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, transcript, locale, intent, raw_data, created_at
            FROM voice_notes
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            if item.get("raw_data"):
                try:
                    item["data"] = json.loads(item["raw_data"])
                except Exception:
                    item["data"] = None
            result.append(item)
        return result


def search_notes(query_text: str, limit: int = 6) -> list[dict]:
    """
    Search past notes using both FTS5 full-text matching and substring/fuzzy fallback.
    Supports Arabic normalized matching.
    """
    if not query_text:
        return []

    norm_query = arabic_norm.normalize_for_search(query_text)
    clean_words = [w for w in norm_query.split() if len(w) > 1]

    with get_connection() as conn:
        results: dict[int, dict] = {}

        # 1. Try FTS5 search
        fts_query = " OR ".join(f'"{w}"*' for w in clean_words) if clean_words else ""
        if fts_query:
            try:
                fts_rows = conn.execute(
                    """
                    SELECT vn.*, rank
                    FROM notes_fts fts
                    JOIN voice_notes vn ON fts.note_id = vn.id
                    WHERE notes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
                for r in fts_rows:
                    results[r["id"]] = dict(r)
            except Exception:
                pass  # Fall back to LIKE search if FTS query syntax fails

        # 2. Fallback / supplement with normalized LIKE search
        if len(results) < limit:
            like_pattern = f"%{norm_query}%"
            rows = conn.execute(
                """
                SELECT * FROM voice_notes
                WHERE normalized_transcript LIKE ? OR transcript LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (like_pattern, f"%{query_text}%", limit),
            ).fetchall()
            for r in rows:
                if r["id"] not in results:
                    results[r["id"]] = dict(r)

        # 3. If query mentions 'tasks' or 'مهام', also pull recent tasks
        if any(k in norm_query.lower() for k in ("task", "مهام", "مهمه", "شغل", "to-do", "todo")):
            task_rows = conn.execute(
                """
                SELECT vn.* FROM tasks t
                JOIN voice_notes vn ON t.note_id = vn.id
                ORDER BY t.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for r in task_rows:
                if r["id"] not in results:
                    results[r["id"]] = dict(r)

        items = list(results.values())[:limit]
        for it in items:
            if it.get("raw_data"):
                try:
                    it["data"] = json.loads(it["raw_data"])
                except Exception:
                    it["data"] = None
        return items


# Initialize on import
init_db()


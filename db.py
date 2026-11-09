"""
Database layer for Voice-to-Insight Agent.

Uses SQLite with FTS5 for local persistence, task status management,
and fast full-text search for RAG knowledge retrieval.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import arabic_norm

# Override with DB_PATH in production and point it at a persistent volume
# (e.g. /var/data/insights.db); container filesystems are wiped on redeploy.
DB_PATH = os.path.abspath(
    os.getenv("DB_PATH") or os.path.join(os.path.dirname(__file__), "data", "insights.db")
)
DB_DIR = os.path.dirname(DB_PATH)


# Soft-deleted tasks can be restored (the UI's "Undo") for this long, then they are
# purged together with their voice note and search index entry.
UNDO_GRACE = timedelta(minutes=10)

# SQL fragment: hide voice notes whose task has been deleted (alias must be `vn`).
_NOT_DELETED_NOTE = (
    "NOT EXISTS (SELECT 1 FROM tasks dt WHERE dt.note_id = vn.id AND dt.deleted_at IS NOT NULL)"
)


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
                due_date_iso TEXT,
                priority TEXT NOT NULL DEFAULT 'medium',
                status TEXT NOT NULL DEFAULT 'pending',
                notion_page_id TEXT,
                notion_url TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                deleted_at TEXT
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
        # Migrate databases created before soft delete existed.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        if "deleted_at" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN deleted_at TEXT")
        if "due_date_iso" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN due_date_iso TEXT")
    purge_deleted_tasks()


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
    due_date_iso: Optional[str] = None,
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
            INSERT INTO tasks (note_id, title, owner, due_date, due_date_iso, priority, status,
                               notion_page_id, notion_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (note_id, title, owner, due_date, due_date_iso, priority, notion_page_id, notion_url, created_at),
        )
        task_id = cur.lastrowid
        conn.commit()

        return {
            "id": task_id,
            "note_id": note_id,
            "title": title,
            "owner": owner,
            "due_date": due_date,
            "due_date_iso": due_date_iso,
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
                "SELECT * FROM tasks WHERE status = ? AND deleted_at IS NULL ORDER BY id DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE deleted_at IS NULL ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def toggle_task(task_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ? AND deleted_at IS NULL", (task_id,)
        ).fetchone()
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


def _hard_delete_tasks(conn: sqlite3.Connection, where: str, params: tuple) -> int:
    """Delete matching tasks plus the voice notes and search entries they came from,
    so deleted tasks stop showing up in history and RAG answers."""
    rows = conn.execute(f"SELECT id, note_id FROM tasks WHERE {where}", params).fetchall()
    if not rows:
        return 0
    task_ids = [r["id"] for r in rows]
    note_ids = [r["note_id"] for r in rows if r["note_id"] is not None]
    marks = ",".join("?" * len(task_ids))
    conn.execute(f"DELETE FROM tasks WHERE id IN ({marks})", task_ids)
    if note_ids:
        nmarks = ",".join("?" * len(note_ids))
        conn.execute(f"DELETE FROM notes_fts WHERE note_id IN ({nmarks})", note_ids)
        conn.execute(f"DELETE FROM voice_notes WHERE id IN ({nmarks})", note_ids)
    return len(task_ids)


def delete_task(task_id: int) -> bool:
    """Soft delete: the task disappears everywhere but can be restored within UNDO_GRACE."""
    purge_deleted_tasks()
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE tasks SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
            (_now_iso(), task_id),
        )
        conn.commit()
        return cur.rowcount > 0


def restore_task(task_id: int) -> Optional[dict]:
    """Undo a soft delete. Returns the restored task, or None if there is nothing to restore."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE tasks SET deleted_at = NULL WHERE id = ? AND deleted_at IS NOT NULL",
            (task_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            return None
        return dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())


def purge_deleted_tasks() -> int:
    """Permanently remove tasks soft-deleted longer than UNDO_GRACE ago."""
    cutoff = (datetime.now(timezone.utc) - UNDO_GRACE).isoformat(timespec="seconds")
    with get_connection() as conn:
        count = _hard_delete_tasks(
            conn, "deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,)
        )
        conn.commit()
        return count


def clear_completed_tasks() -> int:
    with get_connection() as conn:
        count = _hard_delete_tasks(
            conn, "status = 'completed' AND deleted_at IS NULL", ()
        )
        conn.commit()
        return count


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
            f"""
            SELECT id, transcript, locale, intent, raw_data, created_at
            FROM voice_notes vn
            WHERE {_NOT_DELETED_NOTE}
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
                    f"""
                    SELECT vn.*, rank
                    FROM notes_fts fts
                    JOIN voice_notes vn ON fts.note_id = vn.id
                    WHERE notes_fts MATCH ? AND {_NOT_DELETED_NOTE}
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
                f"""
                SELECT * FROM voice_notes vn
                WHERE (normalized_transcript LIKE ? OR transcript LIKE ?) AND {_NOT_DELETED_NOTE}
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
                WHERE t.deleted_at IS NULL
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


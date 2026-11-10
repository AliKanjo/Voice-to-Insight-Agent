"""
Database layer for Voice-to-Insight Agent.

Uses SQLite with FTS5 for local persistence, task status management,
and fast full-text search for RAG knowledge retrieval.
Supports multi-user authentication, sessions, and strict data isolation.
"""

import hashlib
import json
import os
import re
import secrets
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
    """Initialize SQLite tables, user accounts, and FTS5 search index."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Daily request counters for the public demo's quotas (key e.g. "user:5", "ip:1.2.3.4").
            CREATE TABLE IF NOT EXISTS usage_counters (
                day TEXT NOT NULL,
                key TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (day, key)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS voice_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                transcript TEXT NOT NULL,
                normalized_transcript TEXT NOT NULL,
                locale TEXT,
                intent TEXT NOT NULL,
                raw_data TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
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
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                note_id INTEGER REFERENCES voice_notes(id) ON DELETE SET NULL,
                platform TEXT,
                caption TEXT NOT NULL,
                tone TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
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

        # Migrations for existing databases
        def add_column_if_missing(table: str, column: str, col_type: str):
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

        add_column_if_missing("tasks", "deleted_at", "TEXT")
        add_column_if_missing("tasks", "due_date_iso", "TEXT")
        add_column_if_missing("tasks", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")
        add_column_if_missing("voice_notes", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")
        add_column_if_missing("content_items", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")
        add_column_if_missing("summaries", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")

        # Indexes for fast filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_voice_notes_user ON voice_notes(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")

        # Ensure a default user exists for open/test mode
        conn.execute(
            """
            INSERT OR IGNORE INTO users (id, email, password_hash, salt, display_name, created_at)
            VALUES (1, 'default@voiceagent.local', '', '', 'Default User', ?)
            """,
            (_now_iso(),),
        )

    purge_deleted_tasks()


# --- User & Session Authentication ----------------------------------------
def _hash_password(password: str, salt: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 (100,000 iterations)."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def create_user(
    email: str, password: str, display_name: Optional[str] = None
) -> dict:
    """Register a new user account."""
    clean_email = (email or "").strip().lower()
    if not clean_email or "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        raise ValueError("Please provide a valid email address.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    name = (display_name or "").strip()
    if not name:
        name = clean_email.split("@")[0].capitalize()

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    created_at = _now_iso()

    with get_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO users (email, password_hash, salt, display_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_email, pwd_hash, salt, name, created_at),
            )
            user_id = cur.lastrowid
            conn.commit()
            return {
                "id": user_id,
                "email": clean_email,
                "display_name": name,
                "created_at": created_at,
            }
        except sqlite3.IntegrityError:
            raise ValueError("An account with this email already exists.")


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Verify email & password, returning user dictionary or None."""
    clean_email = (email or "").strip().lower()
    if not clean_email or not password:
        return None

    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash, salt, display_name, created_at FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()
        if not row:
            return None

        candidate_hash = _hash_password(password, row["salt"])
        if secrets.compare_digest(candidate_hash, row["password_hash"]):
            return {
                "id": row["id"],
                "email": row["email"],
                "display_name": row["display_name"],
                "created_at": row["created_at"],
            }
        return None


def create_session(user_id: int, days_valid: int = 30) -> str:
    """Create a persistent session token for a user."""
    token = secrets.token_hex(32)
    created_at = _now_iso()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=days_valid)).isoformat(timespec="seconds")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, user_id, created_at, expires_at),
        )
        conn.commit()
    return token


def get_user_by_session(token: str) -> Optional[dict]:
    """Look up an active user by session token. Cleans up if expired."""
    if not token or len(token) < 16:
        return None

    now = _now_iso()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, u.display_name, u.created_at, s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()

        if not row:
            return None

        if row["expires_at"] < now:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return None

        return {
            "id": row["id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "created_at": row["created_at"],
        }


def delete_session(token: str) -> bool:
    """Log out a user by revoking their session token."""
    if not token:
        return False
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        return cur.rowcount > 0


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, display_name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def _valid_user_id(conn: sqlite3.Connection, user_id: Optional[int]) -> Optional[int]:
    if user_id is None:
        return None
    row = conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
    return user_id if row else None


# --- Voice Note Operations ------------------------------------------------
def save_voice_note(
    transcript: str,
    locale: str,
    intent: str,
    raw_data: dict,
    user_id: Optional[int] = None,
) -> int:
    """Insert a recorded voice note and update the search index."""
    norm = arabic_norm.normalize_for_search(transcript)
    created_at = _now_iso()
    raw_json = json.dumps(raw_data, ensure_ascii=False)

    with get_connection() as conn:
        user_id = _valid_user_id(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO voice_notes (user_id, transcript, normalized_transcript, locale, intent, raw_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, transcript, norm, locale, intent, raw_json, created_at),
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
    user_id: Optional[int] = None,
) -> dict:
    created_at = _now_iso()
    with get_connection() as conn:
        user_id = _valid_user_id(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks (user_id, note_id, title, owner, due_date, due_date_iso, priority, status,
                               notion_page_id, notion_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (user_id, note_id, title, owner, due_date, due_date_iso, priority, notion_page_id, notion_url, created_at),
        )
        task_id = cur.lastrowid
        conn.commit()

        return {
            "id": task_id,
            "user_id": user_id,
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


def get_tasks(
    status: Optional[str] = None, user_id: Optional[int] = None
) -> list[dict]:
    with get_connection() as conn:
        conditions = ["deleted_at IS NULL"]
        params: list[Any] = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM tasks WHERE {where_clause} ORDER BY id DESC",
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]


def toggle_task(task_id: int, user_id: Optional[int] = None) -> Optional[dict]:
    with get_connection() as conn:
        conditions = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [task_id]
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        row = conn.execute(
            f"SELECT * FROM tasks WHERE {' AND '.join(conditions)}", tuple(params)
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


def delete_task(task_id: int, user_id: Optional[int] = None) -> bool:
    """Soft delete: the task disappears everywhere but can be restored within UNDO_GRACE."""
    purge_deleted_tasks()
    with get_connection() as conn:
        conditions = ["id = ?", "deleted_at IS NULL"]
        params: list[Any] = [_now_iso(), task_id]
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        cur = conn.execute(
            f"UPDATE tasks SET deleted_at = ? WHERE {conditions[0]} AND {conditions[1]}" +
            (f" AND {conditions[2]}" if user_id is not None else ""),
            tuple(params),
        )
        conn.commit()
        return cur.rowcount > 0


def restore_task(task_id: int, user_id: Optional[int] = None) -> Optional[dict]:
    """Undo a soft delete. Returns the restored task, or None if there is nothing to restore."""
    with get_connection() as conn:
        conditions = ["id = ?", "deleted_at IS NOT NULL"]
        params: list[Any] = [task_id]
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        cur = conn.execute(
            f"UPDATE tasks SET deleted_at = NULL WHERE {' AND '.join(conditions)}",
            tuple(params),
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


def clear_completed_tasks(user_id: Optional[int] = None) -> int:
    with get_connection() as conn:
        where = "status = 'completed' AND deleted_at IS NULL"
        params: tuple = ()
        if user_id is not None:
            where += " AND user_id = ?"
            params = (user_id,)

        count = _hard_delete_tasks(conn, where, params)
        conn.commit()
        return count


# --- Content Item Operations ----------------------------------------------
def save_content(
    platform: Optional[str],
    caption: str,
    tone: Optional[str],
    note_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> dict:
    created_at = _now_iso()
    with get_connection() as conn:
        user_id = _valid_user_id(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO content_items (user_id, note_id, platform, caption, tone, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, note_id, platform, caption, tone, created_at),
        )
        content_id = cur.lastrowid
        conn.commit()
        return {
            "id": content_id,
            "user_id": user_id,
            "note_id": note_id,
            "platform": platform,
            "caption": caption,
            "tone": tone,
            "created_at": created_at,
        }


# --- Summary Operations ---------------------------------------------------
def save_summary(
    summary_text: str,
    key_points: list[str],
    note_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> dict:
    created_at = _now_iso()
    points_json = json.dumps(key_points, ensure_ascii=False)
    with get_connection() as conn:
        user_id = _valid_user_id(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO summaries (user_id, note_id, summary_text, key_points, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, note_id, summary_text, points_json, created_at),
        )
        summary_id = cur.lastrowid
        conn.commit()
        return {
            "id": summary_id,
            "user_id": user_id,
            "note_id": note_id,
            "summary_text": summary_text,
            "key_points": key_points,
            "created_at": created_at,
        }


# --- History & RAG Retrieval ----------------------------------------------
def get_history(limit: int = 50, user_id: Optional[int] = None) -> list[dict]:
    """Retrieve recent voice notes with their classified data, optionally scoped to user."""
    with get_connection() as conn:
        where = _NOT_DELETED_NOTE
        params: list[Any] = []
        if user_id is not None:
            where += " AND vn.user_id = ?"
            params.append(user_id)

        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT id, user_id, transcript, locale, intent, raw_data, created_at
            FROM voice_notes vn
            WHERE {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
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


def search_notes(
    query_text: str, limit: int = 6, user_id: Optional[int] = None
) -> list[dict]:
    """
    Search past notes using both FTS5 full-text matching and substring/fuzzy fallback.
    Supports Arabic normalized matching, strictly filtered by user_id if provided.
    """
    if not query_text:
        return []

    norm_query = arabic_norm.normalize_for_search(query_text)
    clean_words = [w for w in norm_query.split() if len(w) > 1]

    with get_connection() as conn:
        results: dict[int, dict] = {}
        user_cond = "AND vn.user_id = ?" if user_id is not None else ""

        # 1. Try FTS5 search
        fts_query = " OR ".join(f'"{w}"*' for w in clean_words) if clean_words else ""
        if fts_query:
            try:
                fts_params: list[Any] = [fts_query]
                if user_id is not None:
                    fts_params.append(user_id)
                fts_params.append(limit)

                fts_rows = conn.execute(
                    f"""
                    SELECT vn.*, rank
                    FROM notes_fts fts
                    JOIN voice_notes vn ON fts.note_id = vn.id
                    WHERE notes_fts MATCH ? AND {_NOT_DELETED_NOTE} {user_cond}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    tuple(fts_params),
                ).fetchall()
                for r in fts_rows:
                    results[r["id"]] = dict(r)
            except Exception:
                pass  # Fall back to LIKE search if FTS query syntax fails

        # 2. Fallback / supplement with normalized LIKE search
        if len(results) < limit:
            like_pattern = f"%{norm_query}%"
            like_params: list[Any] = [like_pattern, f"%{query_text}%"]
            if user_id is not None:
                like_params.append(user_id)
            like_params.append(limit)

            rows = conn.execute(
                f"""
                SELECT * FROM voice_notes vn
                WHERE (normalized_transcript LIKE ? OR transcript LIKE ?) AND {_NOT_DELETED_NOTE} {user_cond}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(like_params),
            ).fetchall()
            for r in rows:
                if r["id"] not in results:
                    results[r["id"]] = dict(r)

        # 3. If query mentions 'tasks' or 'مهام', also pull recent tasks
        if any(k in norm_query.lower() for k in ("task", "مهام", "مهمه", "شغل", "to-do", "todo")):
            task_cond = "AND t.user_id = ?" if user_id is not None else ""
            t_params: list[Any] = []
            if user_id is not None:
                t_params.append(user_id)
            t_params.append(limit)

            task_rows = conn.execute(
                f"""
                SELECT vn.* FROM tasks t
                JOIN voice_notes vn ON t.note_id = vn.id
                WHERE t.deleted_at IS NULL {task_cond}
                ORDER BY t.id DESC LIMIT ?
                """,
                tuple(t_params),
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


# --- Demo quotas & retention ---------------------------------------------
def consume_quota(day: str, checks: list[tuple[str, int]]) -> tuple[bool, Optional[str]]:
    """Atomically check every (key, limit) for `day` and, only if all are under their
    limit, add one to each. Returns (allowed, first_key_that_was_full)."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")  # serialize concurrent requests
        for key, limit in checks:
            row = conn.execute(
                "SELECT count FROM usage_counters WHERE day = ? AND key = ?", (day, key)
            ).fetchone()
            if (row["count"] if row else 0) >= limit:
                conn.rollback()
                return False, key
        for key, _ in checks:
            conn.execute(
                """INSERT INTO usage_counters (day, key, count) VALUES (?, ?, 1)
                   ON CONFLICT(day, key) DO UPDATE SET count = count + 1""",
                (day, key),
            )
        conn.commit()
        return True, None
    finally:
        conn.close()


def quota_remaining(day: str, checks: list[tuple[str, int]]) -> int:
    """Smallest number of requests left across the given (key, limit) pairs."""
    with get_connection() as conn:
        left = []
        for key, limit in checks:
            row = conn.execute(
                "SELECT count FROM usage_counters WHERE day = ? AND key = ?", (day, key)
            ).fetchone()
            left.append(max(0, limit - (row["count"] if row else 0)))
        return min(left) if left else 0


def purge_older_than(cutoff_iso: str, keep_usage_since_day: str) -> int:
    """Delete notes, tasks, captions and summaries created before `cutoff_iso`, their
    search entries, expired sessions and old usage counters. Returns notes deleted."""
    with get_connection() as conn:
        for table in ("tasks", "content_items", "summaries"):
            conn.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff_iso,))
        cur = conn.execute("DELETE FROM voice_notes WHERE created_at < ?", (cutoff_iso,))
        conn.execute("DELETE FROM notes_fts WHERE note_id NOT IN (SELECT id FROM voice_notes)")
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now_iso(),))
        conn.execute("DELETE FROM usage_counters WHERE day < ?", (keep_usage_since_day,))
        conn.commit()
        return cur.rowcount

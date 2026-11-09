import io
import sqlite3

import db
import main
import stt
from conftest import llm_router, task_result


def understand(client, text):
    return client.post("/api/understand", json={"transcript": text})


# --- understand + actions --------------------------------------------------
def test_task_flow_persists_iso_due_date(client, fake_llm):
    fake_llm(llm_router(task_result(due_date="before Friday", due_date_iso="2030-01-04")))
    r = understand(client, "send the invoice to Sara before Friday")
    assert r.status_code == 200
    task = r.json()["action"]["task"]
    assert task["due_date"] == "before Friday" and task["due_date_iso"] == "2030-01-04"
    stored = client.get("/api/tasks").json()["tasks"][0]
    assert stored["due_date_iso"] == "2030-01-04"


def test_content_and_summary(client, fake_llm):
    fake_llm(llm_router({"kind": "content", "platform": "linkedin", "caption": "Hi", "tone": "pro"}))
    assert understand(client, "caption please").json()["action"]["type"] == "caption_generated"
    fake_llm(llm_router({"kind": "summary", "summary_text": "s", "key_points": ["a"]}))
    assert understand(client, "summarize").json()["action"]["type"] == "summary_shown"


def test_empty_transcript_400(client):
    r = understand(client, "  ")
    assert r.status_code == 400 and r.json()["intent"] == "error"


def test_backend_failure_502_with_generic_message(client, fake_llm):
    def boom(kw):
        raise RuntimeError("secret-internal-detail")

    fake_llm(boom)
    r = understand(client, "hello")
    assert r.status_code == 502
    assert "secret-internal-detail" not in r.text


def test_error_details_exposed_when_enabled(client, fake_llm, monkeypatch):
    monkeypatch.setenv("EXPOSE_ERROR_DETAILS", "true")

    def boom(kw):
        raise RuntimeError("secret-internal-detail")

    fake_llm(boom)
    assert "secret-internal-detail" in understand(client, "hello").text


# --- tasks: delete / restore / clear --------------------------------------
def make_task(client, fake_llm, title="buy zebra food"):
    fake_llm(llm_router(task_result(title=title)))
    return understand(client, title).json()["action"]["task"]["id"]


def test_delete_missing_task_404(client):
    assert client.delete("/api/tasks/999").status_code == 404


def test_soft_delete_hides_task_everywhere_and_restore_brings_it_back(client, fake_llm):
    tid = make_task(client, fake_llm)
    assert client.delete(f"/api/tasks/{tid}").status_code == 200

    assert client.get("/api/tasks").json()["tasks"] == []
    assert "zebra" not in client.get("/api/history").text
    assert db.search_notes("zebra") == []
    assert client.post(f"/api/tasks/{tid}/toggle").status_code == 404

    assert client.post(f"/api/tasks/{tid}/restore").status_code == 200
    assert len(client.get("/api/tasks").json()["tasks"]) == 1
    assert len(db.search_notes("zebra")) == 1
    assert client.post(f"/api/tasks/{tid}/restore").status_code == 404


def test_purge_after_grace_period(client, fake_llm, monkeypatch):
    from datetime import timedelta

    tid = make_task(client, fake_llm)
    client.delete(f"/api/tasks/{tid}")
    monkeypatch.setattr(db, "UNDO_GRACE", timedelta(seconds=-1))
    assert db.purge_deleted_tasks() == 1
    assert client.post(f"/api/tasks/{tid}/restore").status_code == 404
    assert client.get("/api/history").json()["history"] == []


def test_clear_completed_removes_note_from_history(client, fake_llm):
    tid = make_task(client, fake_llm)
    client.post(f"/api/tasks/{tid}/toggle")
    assert client.post("/api/tasks/clear-completed").json()["cleared_count"] == 1
    assert client.get("/api/history").json()["history"] == []


def test_history_limit_bounds(client):
    assert client.get("/api/history?limit=-1").status_code == 422
    assert client.get("/api/history?limit=500").status_code == 422
    assert client.get("/api/history?limit=5").status_code == 200


# --- RAG -------------------------------------------------------------------
def test_rag_includes_iso_due_date_in_context(client, fake_llm):
    make_task(client, fake_llm, title="pay rent")
    seen = {}

    def answer(kw):
        seen["prompt"] = kw["messages"][-1]["content"]
        return "Pay rent on 2030-01-02."

    fake_llm(llm_router(task_result(), answer=answer))
    r = client.post("/api/rag/ask", json={"question": "when is rent due?"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert "2030-01-02" in seen["prompt"]


def test_rag_empty_question_400(client):
    assert client.post("/api/rag/ask", json={"question": " "}).status_code == 400


def test_rag_failure_502(client, fake_llm):
    def boom(kw):
        raise RuntimeError("down")

    fake_llm(boom)
    assert client.post("/api/rag/ask", json={"question": "anything"}).status_code == 502


# --- uploads ---------------------------------------------------------------
def upload(client, name, data, ctype):
    return client.post("/api/transcribe", files={"audio": (name, io.BytesIO(data), ctype)})


def test_upload_rejects_non_audio(client):
    assert upload(client, "notes.txt", b"hi", "text/plain").status_code == 415


def test_upload_rejects_empty(client):
    assert upload(client, "a.wav", b"", "audio/wav").status_code == 400


def test_upload_rejects_oversized(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")  # ~1 KB
    assert upload(client, "a.wav", b"\0" * 200_000, "audio/wav").status_code == 413


def test_transcribe_success_and_no_speech(client, monkeypatch):
    monkeypatch.setattr(stt, "transcribe", lambda path: {"transcript": "hello", "locale": "English"})
    r = upload(client, "a.webm", b"\0" * 100, "video/webm")
    assert r.status_code == 200 and r.json()["transcript"] == "hello"

    def no_speech(path):
        raise stt.NoSpeechError("no speech")

    monkeypatch.setattr(stt, "transcribe", no_speech)
    assert upload(client, "a.wav", b"\0" * 100, "audio/wav").status_code == 422


def test_transcribe_backend_failure_is_generic_502(client, monkeypatch):
    def fail(path):
        raise stt.STTError("secret-internal-detail")

    monkeypatch.setattr(stt, "transcribe", fail)
    r = upload(client, "a.wav", b"\0" * 100, "audio/wav")
    assert r.status_code == 502 and "secret-internal-detail" not in r.text


# --- health ----------------------------------------------------------------
def test_health_ignores_placeholder_keys(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "your-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-real-looking")
    body = client.get("/api/health").json()
    assert body["openai_configured"] is False
    assert body["gemini_configured"] is True


# --- migrations ------------------------------------------------------------
def test_migrates_database_from_older_schema(tmp_path, monkeypatch):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, note_id INTEGER, title TEXT NOT NULL,
           owner TEXT, due_date TEXT, priority TEXT NOT NULL DEFAULT 'medium',
           status TEXT NOT NULL DEFAULT 'pending', notion_page_id TEXT, notion_url TEXT,
           created_at TEXT NOT NULL, completed_at TEXT)"""
    )
    conn.execute("INSERT INTO tasks (title, created_at) VALUES ('old', '2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    cols = {r["name"] for r in db.get_connection().execute("PRAGMA table_info(tasks)")}
    assert {"deleted_at", "due_date_iso"} <= cols
    assert db.get_tasks()[0]["title"] == "old"


def test_frontend_has_no_inline_copy_handlers():
    """Regression for the copy-button XSS: model output must never go into onclick."""
    with open(main.os.path.join(main.STATIC_DIR, "app.js"), encoding="utf-8") as f:
        js = f.read()
    assert 'onclick="copyText' not in js
    assert "escapeAttr" not in js

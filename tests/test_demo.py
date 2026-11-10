"""Public demo mode: sign-in required, daily caps on the owner's key, retention, Notion off."""

import base64
import io

import pytest

import db
import demo
import stt
from conftest import llm_router, task_result


@pytest.fixture
def demo_on(monkeypatch, fake_llm):
    monkeypatch.setenv("DEMO_MODE", "true")
    for name, value in {
        "DEMO_DAILY_PER_USER": "3",
        "DEMO_DAILY_PER_IP": "100",
        "DEMO_DAILY_TOTAL": "100",
        "DEMO_SIGNUPS_PER_IP": "100",
        "DEMO_TRANSCRIBE_DAILY_PER_USER": "100",
    }.items():
        monkeypatch.setenv(name, value)
    return fake_llm(llm_router(task_result(), answer="An answer."))


def signup(client, email):
    r = client.post("/api/auth/register", json={"email": email, "password": "password123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def ask(client, headers, text="call Ahmed tomorrow"):
    return client.post("/api/understand", json={"transcript": text}, headers=headers)


def test_demo_requires_sign_in(client, demo_on):
    r = ask(client, {})
    assert r.status_code == 401
    assert "sign in" in r.json()["error"].lower()
    assert client.get("/api/tasks").status_code == 401
    assert client.post("/api/rag/ask", json={"question": "hi"}).status_code == 401
    assert demo_on.calls == []  # the owner's key was never used
    assert client.get("/").status_code == 200  # the page itself (with the sign-in box) loads


def test_per_user_daily_cap_counts_understand_and_rag(client, demo_on):
    alice = signup(client, "alice@x.io")
    assert ask(client, alice).status_code == 200
    assert client.post("/api/rag/ask", json={"question": "what's next?"}, headers=alice).status_code == 200
    assert ask(client, alice).status_code == 200
    calls_before = len(demo_on.calls)

    r = ask(client, alice)
    assert r.status_code == 429
    assert "Come back tomorrow" in r.json()["error"]
    assert int(r.headers["Retry-After"]) > 0
    assert len(demo_on.calls) == calls_before  # refused before reaching the model

    bob = signup(client, "bob@x.io")  # other accounts are unaffected
    assert ask(client, bob).status_code == 200


def test_ip_cap_spans_accounts(client, demo_on, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_PER_IP", "2")
    a, b = signup(client, "a@x.io"), signup(client, "b@x.io")
    assert ask(client, a).status_code == 200
    assert ask(client, b).status_code == 200
    r = ask(client, signup(client, "c@x.io"))
    assert r.status_code == 429 and "network" in r.json()["error"]


def test_total_cap(client, demo_on, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_TOTAL", "1")
    assert ask(client, signup(client, "a@x.io")).status_code == 200
    r = ask(client, signup(client, "b@x.io"))
    assert r.status_code == 429 and "overall limit" in r.json()["error"]


def test_zero_cap_blocks_everything(client, demo_on, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_TOTAL", "0")
    assert ask(client, signup(client, "a@x.io")).status_code == 429
    assert demo_on.calls == []


def test_empty_input_is_not_charged(client, demo_on):
    alice = signup(client, "alice@x.io")
    for _ in range(5):
        assert ask(client, alice, "  ").status_code == 400
    assert ask(client, alice).status_code == 200


def test_closed_demo(client, demo_on, monkeypatch):
    alice = signup(client, "alice@x.io")
    monkeypatch.setenv("DEMO_CLOSED", "true")
    r = ask(client, alice)
    assert r.status_code == 503 and "ended" in r.json()["error"]
    status = client.get("/api/demo/status", headers=alice).json()
    assert status["closed"] is True and status["remaining_today"] == 0
    assert client.get("/api/tasks", headers=alice).status_code == 200  # still readable


def test_signups_capped_per_ip(client, demo_on, monkeypatch):
    monkeypatch.setenv("DEMO_SIGNUPS_PER_IP", "2")
    signup(client, "a@x.io")
    signup(client, "b@x.io")
    r = client.post("/api/auth/register", json={"email": "c@x.io", "password": "password123"})
    assert r.status_code == 429 and "accounts" in r.json()["error"]


def test_transcription_cap(client, demo_on, monkeypatch):
    monkeypatch.setenv("DEMO_TRANSCRIBE_DAILY_PER_USER", "1")
    monkeypatch.setattr(stt, "transcribe", lambda path: {"transcript": "hi", "locale": "English"})
    alice = signup(client, "alice@x.io")

    def upload():
        files = {"audio": ("a.wav", io.BytesIO(b"\0" * 100), "audio/wav")}
        return client.post("/api/transcribe", files=files, headers=alice)

    assert upload().status_code == 200
    r = upload()
    assert r.status_code == 429 and "transcriptions" in r.json()["error"]


def test_status_endpoint(client, demo_on):
    anon = client.get("/api/demo/status").json()
    assert anon["demo"] is True and "remaining_today" not in anon
    alice = signup(client, "alice@x.io")
    assert client.get("/api/demo/status", headers=alice).json()["remaining_today"] == 3
    ask(client, alice)
    assert client.get("/api/demo/status", headers=alice).json()["remaining_today"] == 2


def test_status_when_not_demo(client):
    assert client.get("/api/demo/status").json() == {"demo": False}


def test_notion_off_in_demo(client, demo_on, monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "secret_x")
    monkeypatch.setenv("NOTION_DATABASE_ID", "db1")
    import actions

    calls = []
    # Record rather than raise: _push_to_notion swallows exceptions, so a raising
    # fake would hide a call.
    monkeypatch.setattr(actions.httpx, "post", lambda *a, **k: calls.append(k))
    r = ask(client, signup(client, "alice@x.io"))
    assert r.status_code == 200
    assert r.json()["action"]["notion"]["pushed"] is False
    assert calls == []


def test_retention_deletes_old_data(client, demo_on):
    alice = signup(client, "alice@x.io")
    assert ask(client, alice).status_code == 200
    with db.get_connection() as conn:
        for table in ("voice_notes", "tasks"):
            conn.execute(f"UPDATE {table} SET created_at = '2000-01-01T00:00:00+00:00'")
        conn.commit()
    demo.maybe_cleanup(force=True)
    assert client.get("/api/tasks", headers=alice).json()["tasks"] == []
    assert client.get("/api/history", headers=alice).json()["history"] == []
    with db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0] == 0


def test_quotas_not_counted_outside_demo(client, fake_llm, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_TOTAL", "0")  # would block everything if demo were on
    fake_llm(llm_router(task_result()))
    assert ask(client, {}).status_code == 200


# --- Auth fixes ---------------------------------------------------------------
def basic(user, pwd):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()}


def test_private_instance_disables_signup(client, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    r = client.post("/api/auth/register", json={"email": "intruder@x.io", "password": "password123"})
    assert r.status_code == 403
    assert "token" not in r.json()


def test_owner_basic_auth_sees_own_tasks(client, fake_llm, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    fake_llm(llm_router(task_result(title="owner task")))
    owner = basic("admin", "s3cret")
    assert client.post("/api/understand", json={"transcript": "x"}, headers=owner).status_code == 200
    assert [t["title"] for t in client.get("/api/tasks", headers=owner).json()["tasks"]] == ["owner task"]


def test_only_configured_username_accepted(client, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.setenv("APP_USERNAME", "ali")
    assert client.get("/api/tasks", headers=basic("admin", "s3cret")).status_code == 401
    assert client.get("/api/tasks", headers=basic("ali", "s3cret")).status_code == 200


def test_login_is_rate_limited(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    codes = [
        client.post("/api/auth/login", json={"email": "a@x.io", "password": f"guess{i}"}).status_code
        for i in range(4)
    ]
    assert codes == [401, 401, 401, 429]

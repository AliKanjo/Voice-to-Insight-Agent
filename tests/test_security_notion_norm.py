import base64
from types import SimpleNamespace

import pytest

import actions
import arabic_norm


# --- auth & rate limiting ---------------------------------------------------
def basic(user, pwd):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()}


def test_auth_required_when_password_set(client, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    assert client.get("/").status_code == 401
    assert client.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks", headers=basic("admin", "wrong")).status_code == 401
    assert client.get("/api/tasks", headers=basic("admin", "s3cret")).status_code == 200
    assert client.get("/healthz").status_code == 200  # liveness probe stays open


def test_open_when_no_password(client):
    assert client.get("/api/tasks").status_code == 200


def test_rate_limit(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    codes = [client.post("/api/understand", json={"transcript": " "}).status_code for _ in range(3)]
    assert codes == [400, 400, 429]
    assert client.get("/api/tasks").status_code == 200  # reads are not limited


# --- Notion -------------------------------------------------------------------
TASK = {"title": "Send invoice", "priority": "high", "due_date_iso": "2030-01-04"}


@pytest.fixture
def notion(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "secret_x")
    monkeypatch.setenv("NOTION_DATABASE_ID", "db123")
    calls = []

    def install(statuses):
        def fake_post(url, headers, json, timeout):
            calls.append(json["properties"])
            code = statuses[min(len(calls), len(statuses)) - 1]
            return SimpleNamespace(
                status_code=code, text="validation error", json=lambda: {"id": "abc-123"}
            )

        monkeypatch.setattr(actions.httpx, "post", fake_post)
        return calls

    return install


def test_notion_sends_due_date_and_priority(notion):
    calls = notion([200])
    out = actions._push_to_notion(TASK)
    assert out["pushed"] and out["url"] == "https://notion.so/abc123"
    assert calls[0]["Due Date"] == {"date": {"start": "2030-01-04"}}
    assert calls[0]["Priority"] == {"select": {"name": "High"}}


def test_notion_falls_back_when_columns_missing(notion):
    calls = notion([400, 400, 200])
    assert actions._push_to_notion(TASK)["pushed"]
    assert ["Due Date" in c for c in calls] == [True, False, False]
    assert ["Priority" in c for c in calls] == [True, True, False]


def test_notion_does_not_retry_auth_errors(notion):
    calls = notion([401])
    out = actions._push_to_notion(TASK)
    assert not out["pushed"] and len(calls) == 1
    assert "validation error" not in out["reason"]  # details stay in the log


def test_notion_custom_property_names_and_no_date(notion, monkeypatch):
    monkeypatch.setenv("NOTION_DUE_PROPERTY", "Deadline")
    calls = notion([200])
    actions._push_to_notion({**TASK, "due_date_iso": None})
    assert "Deadline" not in calls[0] and "Due Date" not in calls[0]


def test_notion_not_configured(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    assert actions._push_to_notion(TASK)["pushed"] is False


# --- Arabic normalization ----------------------------------------------------
def test_normalize_for_search():
    assert arabic_norm.normalize_for_search("أحمد") == arabic_norm.normalize_for_search("احمد")
    assert arabic_norm.normalize_for_search("مدرسة") == "مدرسه"
    assert arabic_norm.normalize_for_search("إلى") == "الي"
    assert arabic_norm.normalize_for_search("٣ مهام") == "3 مهام"


def test_clean_transcript_keeps_letters_and_strips_marks():
    assert arabic_norm.clean_transcript("مُدَرِّسَة  ـــ أحمد ,") == "مدرسة أحمد,"

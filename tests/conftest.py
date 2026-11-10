"""
Shared fixtures. No test touches the network or a real model:
- every test gets a fresh SQLite database in a temp dir,
- `fake_llm` replaces the OpenAI-compatible client used by understand.py and rag.py.
"""

import json
import os
from types import SimpleNamespace

import pytest

# Pin configuration before the app is imported. load_dotenv() never overrides
# variables that are already set, so a developer's .env cannot leak into tests.
os.environ.update(
    {
        "APP_PASSWORD": "",
        "RATE_LIMIT_PER_MINUTE": "0",
        "EXPOSE_ERROR_DETAILS": "false",
        "UNDERSTAND_BACKEND": "local",
        "TRANSCRIBE_BACKEND": "local",
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "",
        "GEMINI_FALLBACK_MODEL": "",
        "NOTION_API_KEY": "",
        "NOTION_DATABASE_ID": "",
        "APP_TIMEZONE": "UTC",
        "MAX_UPLOAD_MB": "25",
        "APP_USERNAME": "admin",
        "DEMO_MODE": "false",
        "DEMO_CLOSED": "false",
    }
)

import db  # noqa: E402
import main  # noqa: E402
import security  # noqa: E402
import understand  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    path = str(tmp_path / "insights.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(db, "DB_DIR", str(tmp_path))
    db.init_db()
    # Rate limiter counters are process-global; start every test clean.
    monkeypatch.setattr(security, "_limiter", security._RateLimiter())
    return path


@pytest.fixture
def client():
    return TestClient(main.app)


class FakeLLM:
    """Stands in for the OpenAI SDK client. `handler(kwargs)` returns either a dict
    (sent back as the JSON content of a structured reply), a str (plain content),
    or raises to simulate an API error."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.handler(kwargs)
        content = json.dumps({"result": out}) if isinstance(out, dict) else out
        message = SimpleNamespace(content=content, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_llm(monkeypatch):
    """Usage: llm = fake_llm(handler). Returns the FakeLLM so tests can inspect calls."""

    def install(handler):
        llm = FakeLLM(handler)
        monkeypatch.setattr(understand, "_resolve_client", lambda: (llm, "fake-model"))
        return llm

    return install


def task_result(**overrides):
    base = {
        "kind": "task",
        "title": "Call Ahmed",
        "owner": None,
        "due_date": "tomorrow",
        "due_date_iso": "2030-01-02",
        "priority": "high",
    }
    base.update(overrides)
    return base


def llm_router(structured, answer="ANSWER"):
    """Handler: structured-output calls get `structured`, plain RAG calls get `answer`."""

    def handler(kwargs):
        if "response_format" in kwargs:
            return structured(kwargs) if callable(structured) else structured
        return answer(kwargs) if callable(answer) else answer

    return handler

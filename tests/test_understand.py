from datetime import date

import pytest

import understand
from conftest import task_result


def test_task_is_parsed(fake_llm):
    fake_llm(lambda kw: task_result())
    out = understand.understand("call Ahmed tomorrow")
    assert out["intent"] == "task"
    assert out["data"]["title"] == "Call Ahmed"
    assert out["data"]["due_date_iso"] == "2030-01-02"


@pytest.mark.parametrize("bad", ["next friday", "2030-13-45", "", 20300102, None])
def test_invalid_iso_due_date_is_dropped(fake_llm, bad):
    fake_llm(lambda kw: task_result(due_date_iso=bad))
    assert understand.understand("x")["data"]["due_date_iso"] is None


def test_prompt_contains_today_and_owner_rule(fake_llm, monkeypatch):
    monkeypatch.setattr(understand, "today", lambda: date(2026, 11, 9))
    llm = fake_llm(lambda kw: task_result())
    understand.understand("send the invoice to Sara by Friday")
    system = llm.calls[0]["messages"][0]["content"]
    assert "Current date: 2026-11-09 (Monday)" in system
    assert "NOT the recipient" in system


def test_schema_requires_due_date_iso():
    task_branch = understand.RESPONSE_SCHEMA["properties"]["result"]["anyOf"][0]
    assert "due_date_iso" in task_branch["required"]
    assert set(task_branch["required"]) == set(task_branch["properties"])  # strict mode needs all


def test_empty_transcript_is_bad_input():
    out = understand.understand("   ")
    assert out["intent"] == "error" and out["error_type"] == "bad_input"


def test_backend_failure_is_upstream(fake_llm):
    def boom(kw):
        raise RuntimeError("provider exploded")

    fake_llm(boom)
    out = understand.understand("hello")
    assert out["error_type"] == "upstream"
    assert "provider exploded" in out["error"]


def test_unparseable_model_output(fake_llm):
    fake_llm(lambda kw: "not json")
    out = understand.understand("hello")
    assert out["intent"] == "error" and "parse" in out["error"]


def test_unknown_kind_rejected(fake_llm):
    fake_llm(lambda kw: {"kind": "poem"})
    assert understand.understand("hello")["intent"] == "error"


def test_gemini_fallback_model_on_rate_limit(monkeypatch):
    """A 429 on the primary model is retried once on GEMINI_FALLBACK_MODEL."""
    from conftest import FakeLLM

    def handler(kw):
        if kw["model"] == "primary":
            raise RuntimeError("Error code: 429 RESOURCE_EXHAUSTED")
        return task_result()

    llm = FakeLLM(handler)
    monkeypatch.setenv("UNDERSTAND_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(understand, "_resolve_client", lambda: (llm, "primary"))
    monkeypatch.setattr(understand.time, "sleep", lambda s: None)
    assert understand.understand("x")["intent"] == "task"
    assert [c["model"] for c in llm.calls] == ["primary", "backup"]


def test_clients_have_bounded_timeout(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "42")
    client = understand.make_client(api_key="x", base_url="http://localhost:1/v1")
    assert client.timeout == 42
    assert client.max_retries == 1


def test_today_respects_timezone(monkeypatch):
    monkeypatch.setenv("APP_TIMEZONE", "Not/AZone")
    assert isinstance(understand.today(), date)  # bad zone falls back to UTC

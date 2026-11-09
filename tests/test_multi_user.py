"""
Tests for Multi-User Authentication, Sessions, and Strict Data Isolation.
"""

import pytest
from conftest import llm_router, task_result


def register(client, email, password, display_name=None):
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "display_name": display_name},
    )


def login(client, email, password):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


def test_registration_and_login_flow(client):
    # Register Alice
    r = register(client, "alice@example.com", "password123", "Alice")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["user"]["email"] == "alice@example.com"
    assert data["user"]["display_name"] == "Alice"
    alice_token = data["token"]
    assert len(alice_token) >= 32

    # Check /api/auth/me
    me = client.get("/api/auth/me", headers=auth_header(alice_token))
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "alice@example.com"

    # Duplicate registration fails
    dup = register(client, "alice@example.com", "otherpassword", "Alice Dup")
    assert dup.status_code == 400
    assert "already exists" in dup.json()["error"]

    # Short password fails
    short = register(client, "bob@example.com", "123", "Bob")
    assert short.status_code == 400

    # Login with wrong password fails
    wrong = login(client, "alice@example.com", "wrongpassword")
    assert wrong.status_code == 401

    # Login with correct password succeeds
    good = login(client, "alice@example.com", "password123")
    assert good.status_code == 200
    new_token = good.json()["token"]

    # Logout revokes token
    out = client.post("/api/auth/logout", headers=auth_header(new_token))
    assert out.status_code == 200
    me_revoked = client.get("/api/auth/me", headers=auth_header(new_token))
    assert me_revoked.status_code == 401


def test_user_data_isolation(client, fake_llm):
    # Register two distinct users: Alice and Bob
    res_a = register(client, "alice@test.com", "password123", "Alice").json()
    token_a = res_a["token"]
    res_b = register(client, "bob@test.com", "password123", "Bob").json()
    token_b = res_b["token"]

    # Alice creates a task
    fake_llm(llm_router(task_result(title="Alice Private Task")))
    r_a = client.post(
        "/api/understand",
        json={"transcript": "Alice Private Task"},
        headers=auth_header(token_a),
    )
    assert r_a.status_code == 200
    alice_task_id = r_a.json()["action"]["task"]["id"]

    # Bob creates a task
    fake_llm(llm_router(task_result(title="Bob Secret Project")))
    r_b = client.post(
        "/api/understand",
        json={"transcript": "Bob Secret Project"},
        headers=auth_header(token_b),
    )
    assert r_b.status_code == 200
    bob_task_id = r_b.json()["action"]["task"]["id"]

    # Alice fetches her tasks: should only see Alice Private Task
    alice_tasks = client.get("/api/tasks", headers=auth_header(token_a)).json()["tasks"]
    assert len(alice_tasks) == 1
    assert alice_tasks[0]["title"] == "Alice Private Task"

    # Bob fetches his tasks: should only see Bob Secret Project
    bob_tasks = client.get("/api/tasks", headers=auth_header(token_b)).json()["tasks"]
    assert len(bob_tasks) == 1
    assert bob_tasks[0]["title"] == "Bob Secret Project"

    # Bob cannot toggle or delete Alice's task
    toggle_attempt = client.post(
        f"/api/tasks/{alice_task_id}/toggle", headers=auth_header(token_b)
    )
    assert toggle_attempt.status_code == 404

    delete_attempt = client.delete(
        f"/api/tasks/{alice_task_id}", headers=auth_header(token_b)
    )
    assert delete_attempt.status_code == 404

    # Alice's task is still pending and not touched
    alice_tasks_after = client.get("/api/tasks", headers=auth_header(token_a)).json()["tasks"]
    assert len(alice_tasks_after) == 1
    assert alice_tasks_after[0]["status"] == "pending"

    # Alice's history does not contain Bob's notes
    alice_history = client.get("/api/history", headers=auth_header(token_a)).json()["history"]
    assert any("Alice" in h["transcript"] for h in alice_history)
    assert not any("Bob" in h["transcript"] for h in alice_history)

    # Bob's history does not contain Alice's notes
    bob_history = client.get("/api/history", headers=auth_header(token_b)).json()["history"]
    assert any("Bob" in h["transcript"] for h in bob_history)
    assert not any("Alice" in h["transcript"] for h in bob_history)

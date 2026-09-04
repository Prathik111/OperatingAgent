"""HTTP surface for tasks + health, driven through ``ASGITransport``.

No lifespan runs under ``ASGITransport``, so these exercise the routing, request
validation and error mapping against the in-memory service wired by fixtures —
the real LangGraph orchestrator is never constructed.
"""

from __future__ import annotations


async def test_health_reports_backend_and_tracks(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["repository"] == "memory"
    assert set(body["tracks"]) == {"native", "langgraph"}


async def test_create_task_returns_202(client, task_service):
    resp = await client.post("/tasks", json={"goal": "say hi", "track": "native"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["id"]
    assert body["goal"] == "say hi"
    assert body["track"] == "native"
    assert body["status"] == "pending"
    await task_service.wait_idle()  # drain the scheduled run


async def test_get_task_reflects_completed_run(client, task_service):
    created = await client.post("/tasks", json={"goal": "say hi", "track": "native"})
    task_id = created.json()["id"]
    await task_service.wait_idle()

    resp = await client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert resp.json()["output"] == "done"
    assert resp.json()["final_message"] == "done"
    assert resp.json()["run_id"]


async def test_get_unknown_task_is_404(client):
    resp = await client.get("/tasks/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


async def test_default_track_used_when_omitted(client, task_service):
    # settings fixture defaults the track to native.
    resp = await client.post("/tasks", json={"goal": "no track given"})
    assert resp.status_code == 202
    assert resp.json()["track"] == "native"
    await task_service.wait_idle()


async def test_empty_goal_is_422(client):
    resp = await client.post("/tasks", json={"goal": ""})  # min_length=1
    assert resp.status_code == 422


async def test_unknown_track_in_body_is_422(client):
    # "bogus" is not a member of AgentTrack -> pydantic rejects it before dispatch.
    resp = await client.post("/tasks", json={"goal": "x", "track": "bogus"})
    assert resp.status_code == 422


async def test_resume_task_creates_a_new_attempt(client, task_service, repository):
    created = await client.post("/tasks", json={"goal": "say hi", "track": "native"})
    task_id = created.json()["id"]
    await task_service.wait_idle()

    response = await client.post(f"/tasks/{task_id}/resume", json={})
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    await task_service.wait_idle()

    assert await repository.get_latest_run_status(task_id) is not None
    assert len(repository._runs) == 2


async def test_get_thread_scoped_task_is_returned_with_final_message(client, task_service):
    thread_id = "conversation-thread"
    created = await client.post(
        "/tasks",
        json={"goal": "threaded", "track": "native", "thread_id": thread_id},
    )
    task_id = created.json()["id"]
    await task_service.wait_idle()

    response = await client.get(f"/threads/{thread_id}/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["thread_id"] == thread_id
    assert response.json()["final_message"] == "done"

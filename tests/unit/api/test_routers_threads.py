"""HTTP thread listing and per-thread task history."""

from __future__ import annotations


async def test_lists_threads_and_tasks_for_one_thread(client, task_service):
    thread_id = "conversation-1"
    first = await client.post(
        "/tasks",
        json={"goal": "first", "track": "native", "thread_id": thread_id},
    )
    second = await client.post(
        "/tasks",
        json={"goal": "second", "track": "langgraph", "thread_id": thread_id},
    )
    await task_service.wait_idle()

    threads_response = await client.get("/threads")
    assert threads_response.status_code == 200
    thread = next(item for item in threads_response.json() if item["id"] == thread_id)
    assert thread["task_count"] == 2

    tasks_response = await client.get(f"/threads/{thread_id}/tasks")
    assert tasks_response.status_code == 200
    tasks = tasks_response.json()
    assert [task["id"] for task in tasks] == [
        second.json()["id"],
        first.json()["id"],
    ]
    assert [task["status"] for task in tasks] == ["completed", "completed"]
    assert [task["output"] for task in tasks] == ["done", "done"]

    nested = await client.get(f"/threads/{thread_id}/tasks/{first.json()['id']}")
    assert nested.status_code == 200
    assert nested.json()["output"] == "done"

    mismatch = await client.get(
        f"/threads/other-thread/tasks/{first.json()['id']}"
    )
    assert mismatch.status_code == 404
    assert "does not belong" in mismatch.json()["detail"]


async def test_unknown_thread_is_404(client):
    response = await client.get("/threads/does-not-exist/tasks")
    assert response.status_code == 404
    assert "thread 'does-not-exist' not found" == response.json()["detail"]


async def test_thread_list_pagination_is_bounded(client):
    assert (await client.get("/threads?limit=0")).status_code == 422
    assert (await client.get("/threads?limit=501")).status_code == 422
    assert (await client.get("/threads?offset=-1")).status_code == 422


async def test_lists_durable_events_for_a_thread(client, task_service):
    thread_id = "conversation-events"
    await client.post(
        "/tasks",
        json={"goal": "first", "track": "native", "thread_id": thread_id},
    )
    await task_service.wait_idle()

    response = await client.get(f"/threads/{thread_id}/events")

    assert response.status_code == 200
    assert [event["type"] for event in response.json()] == ["state", "finished"]

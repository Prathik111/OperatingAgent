"""SSE streaming endpoint, driven through ``ASGITransport``.

The broker topic is pre-populated and closed before the request, so the stream
replays the buffered history and terminates — giving a finite response body the
test client can read in full.
"""

from __future__ import annotations

from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent

from tests.support.langgraph import build_agent_config


async def test_sse_replays_history_and_terminates(client, repository, broker):
    task = AgentTask(id="t-sse", goal="g", thread_id="th", track=AgentTrack.NATIVE)
    await repository.save_task(task)
    await broker.publish("t-sse", AgentEvent("state", {"i": 0}))
    await broker.publish("t-sse", AgentEvent("finished", {"status": "completed"}))
    await broker.close("t-sse")

    resp = await client.get("/tasks/t-sse/events")
    assert resp.status_code == 200
    body = resp.text
    assert "event: state" in body
    assert "event: finished" in body

    scoped = await client.get("/threads/th/tasks/t-sse/events")
    assert scoped.status_code == 200
    assert "event: state" in scoped.text


async def test_sse_rejects_a_task_from_another_thread(client, repository):
    task = AgentTask(id="t-scoped", goal="g", thread_id="th-a", track=AgentTrack.NATIVE)
    await repository.save_task(task)

    response = await client.get("/threads/th-b/tasks/t-scoped/events")

    assert response.status_code == 404


async def test_sse_unknown_task_is_404(client):
    resp = await client.get("/tasks/nope/events")
    assert resp.status_code == 404


async def test_sse_hydrates_events_from_repository_after_restart(
    client, app, repository, settings, background
):
    task = AgentTask(id="t-persisted", goal="g", thread_id="th", track=AgentTrack.NATIVE)
    await repository.save_task(task)
    run_id = await repository.create_run(task.id, build_agent_config())
    await repository.append_event(run_id, AgentEvent("state", {"i": 0}), 0)
    await repository.append_event(run_id, AgentEvent("finished", {"status": "completed"}), 1)
    await repository.finalize_run(
        run_id,
        AgentRunResult(
            status=RunStatus.COMPLETED,
            output="done",
            duration_ms=0,
            llm_calls=0,
            tool_calls=0,
            total_tokens=0,
        ),
    )

    # Replace the in-memory broker topic with a fresh one through the service
    # path used after an API process restart.
    from api.services.event_broker import EventBroker
    from api.services.task_service import TaskService

    service = TaskService(
        orchestrators={},
        repository=repository,
        broker=EventBroker(),
        settings=settings,
        background=background,
    )
    app.state.task_service = service
    response = await client.get("/tasks/t-persisted/events")

    assert response.status_code == 200
    assert "event: state" in response.text
    assert "event: finished" in response.text

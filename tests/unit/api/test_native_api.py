"""Hermetic coverage for the native session-oriented HTTP API."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from agent_native.config import AgentConfig
from agent_native.database import MemoryDatabase
from agent_native.events import EventType
from agent_native.loop import Cancellation, RunRecord, RunStatus
from agent_native.permissions import PermissionDuration, PermissionRequest
from agent_native.service import AgentRuntime, AgentService
from api import create_app
from api.config import ApiSettings
from api.native.dependencies import (
    get_native_runtime,
    get_native_service,
    get_native_settings,
)

from tests._scripted import ScriptedProvider, scripted_registry, text_event


@pytest.fixture
async def native_client() -> AsyncIterator[tuple[httpx.AsyncClient, AgentService, AgentRuntime]]:
    database = MemoryDatabase()
    runtime = AgentRuntime(
        database=database,
        model_registry=scripted_registry(
            provider=ScriptedProvider([text_event("native answer")])
        ),
        agents=[AgentConfig(name="build", model="scripted-1")],
    )
    service = AgentService(runtime)
    settings = ApiSettings(repository_backend="memory")
    app = create_app(settings)
    app.state.native_runtime = runtime
    app.state.native_service = service
    app.state.settings = settings
    app.dependency_overrides[get_native_runtime] = lambda: runtime
    app.dependency_overrides[get_native_service] = lambda: service
    app.dependency_overrides[get_native_settings] = lambda: settings

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client, service, runtime


async def test_native_session_lifecycle(native_client) -> None:
    client, _service, _runtime = native_client

    created = await client.post(
        "/native/sessions",
        json={"agent": "build", "title": "Native", "working_directory": "."},
    )
    assert created.status_code == 201
    session_id = created.json()["id"]

    listed = await client.get("/native/sessions")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == session_id

    detail = await client.get(f"/native/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["message_count"] == 1

    conversation = await client.get(f"/native/sessions/{session_id}/conversation")
    assert conversation.status_code == 200
    assert conversation.json()["messages"][0]["role"] == "system"

    deleted = await client.delete(f"/native/sessions/{session_id}")
    assert deleted.status_code == 204
    assert (await client.get(f"/native/sessions/{session_id}")).status_code == 404


async def test_native_health_and_event_replay(native_client) -> None:
    client, service, runtime = native_client
    session = await service.create_session(agent="build")
    await runtime.events.emit(
        session.id,
        EventType.MESSAGE_ADDED,
        {"id": "message-1", "role": "user"},
        run_id="run-1",
    )

    health = await client.get("/native/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "scripted-1" in health.json()["models"]

    events = await client.get(f"/native/sessions/{session.id}/events")
    assert events.status_code == 200
    assert events.json()[0]["type"] == EventType.MESSAGE_ADDED
    assert events.json()[0]["run_id"] == "run-1"


async def test_native_message_stream_completes(native_client) -> None:
    client, service, _runtime = native_client
    session = await service.create_session(agent="build")

    response = await client.post(
        f"/native/sessions/{session.id}/messages",
        json={"message": "hello"},
    )
    assert response.status_code == 200
    assert "native answer" in response.text
    assert "run_finished" in response.text
    assert "run_receipt" in response.text
    assert "final_message" in response.text
    runs = await client.get(f"/native/sessions/{session.id}/runs")
    assert runs.status_code == 200
    assert runs.json()[0]["final_text"] == "native answer"
    assert runs.json()[0]["final_message"] == "native answer"


async def test_native_message_validation_and_missing_session(native_client) -> None:
    client, _service, _runtime = native_client

    empty = await client.post("/native/sessions/missing/messages", json={})
    assert empty.status_code == 422

    missing = await client.post(
        "/native/sessions/missing/messages", json={"message": "hello"}
    )
    assert missing.status_code == 404

    resume = await client.post("/native/sessions/missing/resume", json={})
    assert resume.status_code == 404

    cancel = await client.post("/native/sessions/missing/cancel")
    assert cancel.status_code == 404


async def test_native_run_endpoints_return_records_and_not_found(native_client) -> None:
    client, service, runtime = native_client
    session = await service.create_session(agent="build")
    record = RunRecord(
        run_id="run-record-1",
        session_id=session.id,
        status=RunStatus.FINISHED.value,
        turns=2,
        input_tokens=5,
        output_tokens=3,
        model="scripted-1",
    )
    await runtime.database.save_run(record)

    detail = await client.get("/native/runs/run-record-1")
    assert detail.status_code == 200
    assert detail.json()["run_id"] == "run-record-1"
    assert detail.json()["status"] == "finished"
    assert detail.json()["input_tokens"] == 5
    assert detail.json()["final_message"] == ""

    listed = await client.get(f"/native/sessions/{session.id}/runs")
    assert listed.status_code == 200
    assert [item["run_id"] for item in listed.json()] == ["run-record-1"]
    assert listed.json()[0]["final_text"] == ""
    assert listed.json()[0]["final_message"] == ""

    assert (await client.get("/native/runs/missing")).status_code == 404
    assert (await client.get("/native/sessions/missing/runs")).status_code == 404


async def test_native_permission_endpoints_list_get_and_resolve(native_client) -> None:
    client, service, _runtime = native_client
    request = PermissionRequest(
        call_id="call-1",
        tool="filesystem.write_file",
        arguments={"path": "notes/today.md"},
        preview="write notes/today.md",
        reason="mutating tool",
    )
    service.pending_permissions = lambda: [request]
    resolved: dict = {}

    async def resolve_permission(call_id, allowed, duration, scope):
        resolved.update(
            call_id=call_id,
            allowed=allowed,
            duration=duration,
            scope=scope,
        )

    service.resolve_permission = resolve_permission

    listed = await client.get("/native/permissions", params={"session_id": "session-1"})
    assert listed.status_code == 200
    assert listed.json()[0]["call_id"] == "call-1"
    assert listed.json()[0]["tool"] == "filesystem.write_file"

    detail = await client.get("/native/permissions/call-1")
    assert detail.status_code == 200
    assert detail.json()["preview"] == "write notes/today.md"

    decision = await client.post(
        "/native/permissions/call-1",
        json={"allowed": True, "duration": "session", "scope": "notes"},
    )
    assert decision.status_code == 200
    assert decision.json() == {
        "call_id": "call-1",
        "allowed": True,
        "duration": "session",
        "scope": "notes",
    }
    assert resolved == {
        "call_id": "call-1",
        "allowed": True,
        "duration": PermissionDuration.SESSION,
        "scope": "notes",
    }

    assert (await client.get("/native/permissions/missing")).status_code == 404
    assert (
        await client.post(
            "/native/permissions/missing",
            json={"allowed": False},
        )
    ).status_code == 404


async def test_native_resume_and_cancel_lifecycle(native_client) -> None:
    client, service, _runtime = native_client
    session = await service.create_session(agent="build")

    resumed = await client.post(
        f"/native/sessions/{session.id}/resume",
        json={"limits": {"max_turns": 1}},
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "finished"

    idle = await client.post(f"/native/sessions/{session.id}/cancel")
    assert idle.status_code == 202
    assert idle.json()["cancelled"] is False

    cancellation = Cancellation()
    client._transport.app.state.native_cancels = {session.id: cancellation}
    active = await client.post(f"/native/sessions/{session.id}/cancel")
    assert active.status_code == 202
    assert active.json()["cancelled"] is True
    assert cancellation.cancelled is True


async def test_native_dependency_reports_unavailable_without_lifespan() -> None:
    settings = ApiSettings(repository_backend="memory")
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/native/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "native runtime not initialized"

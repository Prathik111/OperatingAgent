"""Hermetic coverage for the native session-oriented HTTP API."""

from __future__ import annotations

import asyncio
import json
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
        session_id="session-1",
    )
    service.pending_permissions = lambda session_id="": [
        pending
        for pending in [request]
        if not session_id or pending.session_id == session_id
    ]
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
    client._transport.app.state.native_cancels = {"run_test_1": cancellation}
    client._transport.app.state.native_cancel_sessions = {session.id: {"run_test_1"}}
    active = await client.post(
        f"/native/sessions/{session.id}/cancel", params={"run_id": "run_test_1"}
    )
    assert active.status_code == 202
    assert active.json() == {
        "session_id": session.id,
        "run_id": "run_test_1",
        "cancelled": True,
    }
    assert cancellation.cancelled is True


class GatedProvider:
    """Blocks its first stream call on a gate; later calls answer at once.

    Lets a test park run A mid-turn while run B completes, so cancellation
    isolation is exercised deterministically instead of by timing luck.
    """

    def __init__(self, gate: asyncio.Event) -> None:
        self._gate = gate
        self._calls = 0

    async def stream(self, messages, tools, model, temperature=0.0):
        self._calls += 1
        if self._calls == 1:
            yield text_event("A-part")
            await self._gate.wait()
            yield text_event("A-late")
        else:
            yield text_event("B-done")

    def count_tokens(self, messages) -> int:
        return 0


class SlowProvider:
    """Answers after a short sleep, so concurrent requests genuinely overlap."""

    def __init__(self, delay: float = 0.2) -> None:
        self._delay = delay
        self.requests: list = []

    async def stream(self, messages, tools, model, temperature=0.0):
        self.requests.append((messages, tools))
        await asyncio.sleep(self._delay)
        yield text_event("slow answer")

    def count_tokens(self, messages) -> int:
        return 0


@pytest.fixture
async def gated_native_client() -> AsyncIterator:
    """Native app wired to a gated provider, plus the gate and provider."""
    from agent_native.service import AgentRuntime, AgentService

    gate = asyncio.Event()
    provider = GatedProvider(gate)
    database = MemoryDatabase()
    runtime = AgentRuntime(
        database=database,
        model_registry=scripted_registry(provider=provider),
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
        yield client, service, runtime, gate, provider


def _sse_frames(text: str) -> list[dict]:
    frames = []
    for raw in text.replace("\r\n", "\n").split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        frame: dict = {}
        for line in raw.splitlines():
            if line.startswith("id:"):
                frame["id"] = line[3:].strip()
            elif line.startswith("event:"):
                frame["event"] = line[6:].strip()
            elif line.startswith("data:"):
                frame["data"] = line[5:].strip()
        frames.append(frame)
    return frames


def _receipt_frame(text: str) -> dict:
    for frame in _sse_frames(text):
        if frame.get("event") == "run_receipt":
            return json.loads(frame["data"])
    raise AssertionError("no run_receipt frame in SSE stream")


async def _wait_for_event_text(client, session_id: str, text: str, timeout: float = 10.0) -> str:
    """Poll the event log until a delta with the given text lands; return its run id."""
    waited = 0.0
    while waited < timeout:
        events = (await client.get(f"/native/sessions/{session_id}/events")).json()
        for event in events:
            if event["type"] == "assistant_delta" and event["data"].get("text") == text:
                return event["run_id"]
        await asyncio.sleep(0.05)
        waited += 0.05
    raise AssertionError(f"timed out waiting for event text {text!r}")


async def test_message_stream_captures_cursor_before_run_starts(native_client) -> None:
    """Regression (P0-1): the SSE cursor must predate the run.

    Delays every event-log read so the background run emits its whole
    transcript before the endpoint snapshots its cursor. With the old
    snapshot-after-start ordering the stream then misses everything and
    hangs; with the fix it streams every event exactly once plus the receipt.
    """
    client, service, runtime = native_client
    session = await service.create_session(agent="build")

    db = runtime.database
    real_load_events = db.load_events

    async def slow_load_events(session_id, after_sequence=0):
        await asyncio.sleep(0.1)
        return await real_load_events(session_id, after_sequence)

    db.load_events = slow_load_events  # type: ignore[method-assign]
    try:
        response = await asyncio.wait_for(
            client.post(f"/native/sessions/{session.id}/messages", json={"message": "hello"}),
            timeout=10,
        )
    finally:
        db.load_events = real_load_events
    assert response.status_code == 200

    frames = _sse_frames(response.text)
    numeric = [frame for frame in frames if frame.get("id", "").isdigit()]
    sequences = [int(frame["id"]) for frame in numeric]
    # No gaps, no duplicates, starting at the run's first event.
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    assert sequences[0] == 1
    assert numeric[0]["event"] == "message_added"
    assert "run_finished" in [frame.get("event") for frame in frames]
    assert _receipt_frame(response.text)["status"] == "finished"


async def test_cancel_targets_exactly_one_run(gated_native_client) -> None:
    """Regression (P0-2): cancelling run A never touches run B.

    Run A parks mid-turn in the provider gate while run B completes. A broad
    cancel then stops only the still-running run, and run A's registry entry
    is cleaned up so a later cancel reports a clean miss.
    """
    client, service, _runtime, gate, _provider = gated_native_client
    session = await service.create_session(agent="build")

    task_a = asyncio.create_task(
        client.post(f"/native/sessions/{session.id}/messages", json={"message": "A"})
    )
    run_a = await _wait_for_event_text(client, session.id, "A-part")

    response_b = await client.post(f"/native/sessions/{session.id}/messages", json={"message": "B"})
    assert response_b.status_code == 200
    receipt_b = _receipt_frame(response_b.text)
    assert receipt_b["status"] == "finished"
    assert "B-done" in response_b.text
    run_b = receipt_b["run_id"]
    assert run_b != run_a

    # Broad cancel stops the parked run only — never the finished one.
    broad = await client.post(f"/native/sessions/{session.id}/cancel")
    assert broad.status_code == 202
    assert broad.json()["cancelled"] is True
    assert broad.json()["cancelled_runs"] == [run_a]

    gate.set()  # let parked run A observe the cancellation
    response_a = await asyncio.wait_for(task_a, timeout=10)
    assert _receipt_frame(response_a.text)["status"] == "cancelled"

    # Cleanup removed only run A's entry: cancelling it again is a clean miss.
    again = await client.post(
        f"/native/sessions/{session.id}/cancel", params={"run_id": run_a}
    )
    assert again.json() == {
        "session_id": session.id,
        "run_id": run_a,
        "cancelled": False,
        "reason": "no active run",
    }
    missing = await client.post(
        f"/native/sessions/{session.id}/cancel", params={"run_id": "run_nope"}
    )
    assert missing.json()["cancelled"] is False


async def test_concurrent_resumes_run_the_model_once() -> None:
    """Regression (P0-2): racing resumes serialize; the second is a no-op receipt.

    The slow provider keeps the first resume's loop in flight while the second
    resume checks the log, so without serialization both would execute. The
    per-session resume lock means only the first runs the model; the second
    observes the first run's RUN_FINISHED and returns its receipt instead.
    """
    provider = SlowProvider(delay=0.2)
    database = MemoryDatabase()
    runtime = AgentRuntime(
        database=database,
        model_registry=scripted_registry(provider=provider),
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
        session = await service.create_session(agent="build")

        first, second = await asyncio.gather(
            client.post(f"/native/sessions/{session.id}/resume", json={}),
            client.post(f"/native/sessions/{session.id}/resume", json={}),
        )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "finished"
    assert second.json()["status"] == "finished"
    assert first.json()["run_id"] == second.json()["run_id"]
    assert len(provider.requests) == 1


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


@pytest.mark.regression
async def test_native_settings_auto_approve_all_roundtrip(native_client) -> None:
    client, _service, runtime = native_client

    initial = await client.get("/native/settings")
    assert initial.status_code == 200
    assert initial.json()["auto_approve_all"] is False

    # Flags-only patch must not trip LLM provider validation.
    enabled = await client.patch("/native/settings", json={"auto_approve_all": True})
    assert enabled.status_code == 200
    assert enabled.json()["auto_approve_all"] is True
    assert runtime.auto_approve_all is True

    current = await client.get("/native/settings")
    assert current.json()["auto_approve_all"] is True

    disabled = await client.patch("/native/settings", json={"auto_approve_all": False})
    assert disabled.status_code == 200
    assert disabled.json()["auto_approve_all"] is False
    assert runtime.auto_approve_all is False


@pytest.mark.regression
async def test_native_sandbox_reports_live_probe(native_client) -> None:
    client, _service, runtime = native_client

    # No pool configured on the fixture runtime: honest disconnected status.
    missing = await client.get("/native/sandbox")
    assert missing.status_code == 200
    assert missing.json()["available"] is False

    class _Pool:
        image = "test-image:1"
        reason = ""

        def __init__(self) -> None:
            self.probes = 0

        async def probe(self) -> bool:
            self.probes += 1
            return True

        def status_line(self) -> str:
            return "sandbox: on - Docker container (test-image:1)"

    pool = _Pool()
    runtime.sandbox = pool
    try:
        first = await client.get("/native/sandbox")
        assert first.status_code == 200
        assert first.json() == {
            "available": True,
            "image": "test-image:1",
            "status": "sandbox: on - Docker container (test-image:1)",
            "reason": "",
        }
        # Probed fresh on every call: starting Docker flips this with no restart.
        second = await client.get("/native/sandbox")
        assert second.status_code == 200
        assert pool.probes == 2
    finally:
        runtime.sandbox = None

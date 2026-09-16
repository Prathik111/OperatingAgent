"""Restart-style coverage for the desktop SQLite repository."""

from __future__ import annotations

from datetime import UTC

import httpx
from agent_native.conversation import Session, user_message
from agent_native.database import MemoryDatabase
from agent_native.events import Event, EventType
from agent_native.sqlite import SQLiteDatabase
from api.app import create_app
from api.config import ApiSettings
from api.repository.sqlite import SQLiteTaskRepository
from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus


async def test_api_sqlite_repository_survives_reopen(tmp_path) -> None:
    path = tmp_path / "operating-agent.db"
    task = AgentTask(
        id="task-1",
        goal="remember this",
        thread_id="thread-1",
        track=AgentTrack.LANGGRAPH,
    )
    first = SQLiteTaskRepository(path)
    await first.save_task(task)
    run_id = await first.create_run(task.id, ApiSettings().build_agent_config())
    await first.finalize_run(
        run_id,
        AgentRunResult(
            status=RunStatus.COMPLETED,
            output="persisted result",
            duration_ms=1,
            llm_calls=1,
            tool_calls=0,
            total_tokens=3,
        ),
    )
    await first.close()

    second = SQLiteTaskRepository(path)
    assert (await second.get_task(task.id)).goal == task.goal
    receipt = await second.get_latest_run(task.id)
    assert receipt is not None
    assert receipt.output == "persisted result"
    await second.close()


async def test_api_sqlite_repository_normalizes_legacy_timestamps_on_reopen(
    tmp_path,
) -> None:
    path = tmp_path / "legacy.db"
    first = SQLiteTaskRepository(path)
    await first.save_task(
        AgentTask(
            id="legacy-task",
            goal="legacy",
            thread_id="legacy-thread",
            track=AgentTrack.LANGGRAPH,
        )
    )
    first._tasks["legacy-task"].created_at = first._tasks[
        "legacy-task"
    ].created_at.replace(tzinfo=None)
    first._threads["legacy-thread"].created_at = first._threads[
        "legacy-thread"
    ].created_at.replace(tzinfo=None)
    first._threads["legacy-thread"].updated_at = first._threads[
        "legacy-thread"
    ].updated_at.replace(tzinfo=None)
    await first.close()

    second = SQLiteTaskRepository(path)
    task = await second.get_task("legacy-task")
    thread = (await second.list_threads(limit=10, offset=0))[0]

    assert task.created_at.tzinfo is UTC
    assert thread.created_at.tzinfo is UTC
    assert thread.updated_at.tzinfo is UTC
    await second.close()


async def test_native_sqlite_database_survives_reopen(tmp_path) -> None:
    path = tmp_path / "operating-agent.db"
    first = SQLiteDatabase(path)
    session = Session(id="session-1", working_directory=str(tmp_path))
    await first.create_session(session)
    await first.save_message(user_message(session.id, "hello"))
    sequence = await first.next_sequence(session.id)
    await first.save_event(
        Event(sequence, EventType.TURN_STARTED, session.id, data={"text": "hello"})
    )
    await first.close()

    second = SQLiteDatabase(path)
    restored = await second.get_session(session.id)
    assert restored is not None
    assert (await second.load_conversation(session.id)).messages[0].text() == "hello"
    events = await second.load_events(session.id)
    assert len(events) == 1
    assert events[0].sequence == 1
    await second.close()


async def test_native_sqlite_upgrades_sessions_without_timestamps(tmp_path) -> None:
    path = tmp_path / "legacy-native.db"
    first = SQLiteDatabase(path)
    legacy = Session(id="legacy-session", working_directory=str(tmp_path))
    del legacy.created_at
    del legacy.updated_at
    first._sessions[legacy.id] = legacy
    await first.close()

    second = SQLiteDatabase(path)
    restored = await second.get_session(legacy.id)
    assert restored is not None
    assert restored.created_at.tzinfo is UTC
    assert restored.updated_at.tzinfo is UTC
    assert [item.id for item in await second.list_sessions()] == [legacy.id]
    await second.close()


def test_sqlite_settings_propagate_repository_and_checkpoint_path(tmp_path) -> None:
    settings = ApiSettings(
        repository_backend="sqlite",
        sqlite_database_path=str(tmp_path / "agent.db"),
    )
    config = settings.build_agent_config()
    assert config.checkpoint.backend == "sqlite"
    assert config.checkpoint.connection_string == str(tmp_path / "agent.db")


def test_native_sqlite_is_a_database() -> None:
    assert issubclass(SQLiteDatabase, MemoryDatabase)


async def test_api_lifespan_uses_sqlite_for_both_tracks(tmp_path) -> None:
    app = create_app(
        ApiSettings(
            repository_backend="sqlite",
            sqlite_database_path=str(tmp_path / "operating-agent.db"),
            sandbox_enabled=False,
        )
    )
    async with app.router.lifespan_context(app):
        assert type(app.state.repository).__name__ == "SQLiteTaskRepository"
        assert type(app.state.native_runtime.database).__name__ == "SQLiteDatabase"
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/health")
        assert response.status_code == 200

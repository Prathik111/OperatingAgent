"""``InMemoryTaskRepository`` — the default hermetic backend."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from api.errors import TaskNotFound, ThreadNotFound
from api.repository.memory import InMemoryTaskRepository
from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus, TaskStatus
from common.events import AgentEvent

from tests.support.langgraph import build_agent_config


def _task(task_id: str = "task-1") -> AgentTask:
    return AgentTask(id=task_id, goal="g", thread_id="th", track=AgentTrack.NATIVE)


async def test_save_and_get_round_trip():
    repo = InMemoryTaskRepository()
    task = _task()
    await repo.save_task(task)

    assert await repo.get_task("task-1") is task
    assert repo.task_status("task-1") == TaskStatus.PLANNING


async def test_get_unknown_raises_task_not_found():
    repo = InMemoryTaskRepository()
    with pytest.raises(TaskNotFound):
        await repo.get_task("nope")


async def test_run_lifecycle_updates_status_and_orders_events():
    repo = InMemoryTaskRepository()
    task = _task()
    await repo.save_task(task)
    run_id = await repo.create_run(task.id, build_agent_config())

    assert await repo.get_latest_run_status(task.id) == RunStatus.CREATED
    await repo.mark_run_running(run_id)
    assert await repo.get_latest_run_status(task.id) == RunStatus.RUNNING

    await repo.append_event(run_id, AgentEvent("state", {"i": 0}), 0)
    await repo.append_event(run_id, AgentEvent("finished", {"i": 1}), 1)

    result = AgentRunResult(
        status=RunStatus.COMPLETED,
        output="out",
        duration_ms=0.0,
        llm_calls=0,
        tool_calls=0,
        total_tokens=0,
    )
    await repo.finalize_run(run_id, result)
    await repo.update_task_status(task.id, TaskStatus.COMPLETED)

    assert await repo.get_latest_run_status(task.id) == RunStatus.COMPLETED
    assert repo.task_status(task.id) == TaskStatus.COMPLETED
    assert [(seq, typ) for seq, typ, _ in repo.events_for(run_id)] == [
        (0, "state"),
        (1, "finished"),
    ]


async def test_latest_run_status_none_without_runs():
    repo = InMemoryTaskRepository()
    assert await repo.get_latest_run_status("task-1") is None


async def test_lists_threads_and_tasks_newest_first_with_pagination():
    repo = InMemoryTaskRepository()
    first = AgentTask(
        id="task-1",
        goal="first",
        thread_id="thread-a",
        track=AgentTrack.NATIVE,
        metadata={"title": "Thread A"},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = AgentTask(
        id="task-2",
        goal="second",
        thread_id="thread-a",
        track=AgentTrack.NATIVE,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    other = AgentTask(
        id="task-3",
        goal="other",
        thread_id="thread-b",
        track=AgentTrack.LANGGRAPH,
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
    )
    for task in (first, second, other):
        await repo.save_task(task)

    run_id = await repo.create_run(first.id, build_agent_config())
    await repo.mark_run_running(run_id)

    threads = await repo.list_threads(limit=10, offset=0)
    assert [thread.id for thread in threads] == ["thread-b", "thread-a"]
    assert threads[1].title == "Thread A"
    assert threads[1].task_count == 2
    assert [thread.id for thread in await repo.list_threads(limit=1, offset=1)] == [
        "thread-a"
    ]

    tasks = await repo.list_tasks_by_thread("thread-a", limit=10, offset=0)
    assert [task.id for task, _status in tasks] == ["task-2", "task-1"]
    assert [status for _task, status in tasks] == [None, RunStatus.RUNNING]
    paged = await repo.list_tasks_by_thread("thread-a", limit=1, offset=1)
    assert [task.id for task, _status in paged] == ["task-1"]


async def test_list_tasks_for_unknown_thread_raises():
    repo = InMemoryTaskRepository()
    with pytest.raises(ThreadNotFound):
        await repo.list_tasks_by_thread("missing", limit=10, offset=0)

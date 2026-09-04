"""``TaskService`` — non-blocking create, background run, terminal persistence."""

from __future__ import annotations

import pytest
from api.errors import TaskAlreadyRunning, UnknownTrack
from api.services.task_service import TaskService
from common.enums import AgentTrack, RunStatus, TaskStatus


async def test_create_returns_before_run_completes(
    task_service, orchestrator, repository, background
):
    task = await task_service.create_task(goal="do a thing")

    # create_task returns immediately with the task; the run is only scheduled.
    assert task.id
    assert task.track == AgentTrack.NATIVE
    assert len(background) == 1
    assert await repository.get_task(task.id) is task

    await task_service.wait_idle()

    assert orchestrator.seen[0].id == task.id
    assert await repository.get_latest_run_status(task.id) == RunStatus.COMPLETED
    assert repository.task_status(task.id) == TaskStatus.COMPLETED


async def test_events_persisted_in_order_and_replayable(task_service, broker):
    task = await task_service.create_task(goal="stream me")
    await task_service.wait_idle()

    # A late subscriber replays the full sequence; the topic is already closed.
    events = [e async for e in broker.subscribe(task.id)]
    assert [e.type for e in events] == ["state", "finished"]


async def test_failing_orchestrator_marks_run_failed(
    make_orchestrator, repository, broker, settings, background
):
    boom = make_orchestrator(raise_exc=RuntimeError("boom"))
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: boom},
        repository=repository,
        broker=broker,
        settings=settings,
        background=background,
    )

    task = await service.create_task(goal="will fail")
    await service.wait_idle()

    assert await repository.get_latest_run_status(task.id) == RunStatus.FAILED
    assert repository.task_status(task.id) == TaskStatus.FAILED
    # The service synthesises an error event before finalizing the run.
    events = [e async for e in broker.subscribe(task.id)]
    assert "error" in [e.type for e in events]


async def test_unknown_track_rejected(
    orchestrator, repository, broker, settings, background
):
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},  # LANGGRAPH not registered
        repository=repository,
        broker=broker,
        settings=settings,
        background=background,
    )
    with pytest.raises(UnknownTrack):
        await service.create_task(goal="x", track=AgentTrack.LANGGRAPH)


async def test_explicit_thread_id_uses_new_then_continue_mode(
    task_service, orchestrator
):
    first = await task_service.create_task(goal="first", thread_id="conversation-1")
    await task_service.wait_idle()
    second = await task_service.create_task(goal="second", thread_id="conversation-1")
    await task_service.wait_idle()

    assert [task.execution_mode for task in orchestrator.seen] == ["new", "continue"]
    assert first.thread_id == second.thread_id == "conversation-1"


async def test_concurrent_tasks_on_one_thread_are_rejected(task_service):
    await task_service.create_task(goal="first", thread_id="conversation-1")
    with pytest.raises(TaskAlreadyRunning):
        await task_service.create_task(goal="second", thread_id="conversation-1")
    await task_service.wait_idle()

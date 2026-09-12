"""DB-authoritative execution ownership: restart recovery and no duplicates.

The in-memory active-task/thread sets die with the process while durable rows
survive it. These tests prove the database — not process memory — decides what
may execute: concurrent starts/resumes admit exactly one winner, a simulated
restart reaps stale runs as INTERRUPTED with history intact, and a run owned
by somebody else is never touched.
"""

from __future__ import annotations

import asyncio

import pytest
from api.errors import TaskAlreadyRunning
from api.services.task_service import TaskService
from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus, TaskStatus
from common.events import AgentEvent


class GatedOrchestrator:
    """Stays inside ``run`` until released, so tests own the timing."""

    def __init__(self) -> None:
        self.seen: list[AgentTask] = []
        self._gate = asyncio.Event()

    def release(self) -> None:
        self._gate.set()

    async def run(self, task: AgentTask, on_event=None) -> AgentRunResult:
        self.seen.append(task)
        await self._gate.wait()
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output="done",
            duration_ms=0.0,
            llm_calls=0,
            tool_calls=0,
            total_tokens=0,
        )

    async def aclose(self) -> None:
        pass


async def _craft_stale_run(repository, settings, *, owner: str) -> tuple[str, str]:
    """Persist exactly what a crash leaves behind: a RUNNING row, one event."""
    task = AgentTask(
        id="task-stale-1",
        goal="crashed mid-run",
        thread_id="thread-stale",
        track=AgentTrack.NATIVE,
        metadata={},
    )
    await repository.save_task(task)
    run_id = await repository.create_run(
        task.id,
        settings.build_agent_config(AgentTrack.NATIVE),
        metadata={"execution_owner": owner},
    )
    await repository.mark_run_running(run_id)
    await repository.append_event(
        run_id, AgentEvent(type="state", payload={"status": "running"}), 0
    )
    return task.id, run_id


async def test_concurrent_starts_single_winner(task_service, orchestrator):
    results = await asyncio.gather(
        task_service.create_task(goal="a", thread_id="race-thread"),
        task_service.create_task(goal="b", thread_id="race-thread"),
        return_exceptions=True,
    )
    oks = [r for r in results if not isinstance(r, BaseException)]
    errs = [r for r in results if isinstance(r, BaseException)]
    assert len(oks) == 1
    assert len(errs) == 1 and isinstance(errs[0], TaskAlreadyRunning)
    await task_service.wait_idle()
    assert len(orchestrator.seen) == 1


async def test_concurrent_resumes_single_winner(task_service, orchestrator):
    task = await task_service.create_task(goal="once")
    await task_service.wait_idle()
    assert len(orchestrator.seen) == 1

    results = await asyncio.gather(
        task_service.resume_task(task.id),
        task_service.resume_task(task.id),
        return_exceptions=True,
    )
    oks = [r for r in results if not isinstance(r, BaseException)]
    errs = [r for r in results if isinstance(r, BaseException)]
    assert len(oks) == 1
    assert len(errs) == 1 and isinstance(errs[0], TaskAlreadyRunning)
    await task_service.wait_idle()
    assert len(orchestrator.seen) == 2


async def test_restart_recovery_marks_stale_runs_interrupted(
    repository, settings, broker, orchestrator, background
):
    task_id, run_id = await _craft_stale_run(
        repository, settings, owner="dead-process"
    )
    assert await repository.get_latest_run_status(task_id) is RunStatus.RUNNING

    # A new service is a new process: empty registries, fresh owner token.
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=broker,
        settings=settings,
        background=background,
    )
    assert service._active_task_ids == set()

    recovered = await service.recover_stale_executions()
    assert recovered == [run_id]

    summary = await repository.get_latest_run(task_id)
    assert summary is not None
    assert summary.status is RunStatus.INTERRUPTED
    assert summary.metadata["execution_owner"] == "dead-process"
    assert summary.metadata["recovery"]["previous_status"] == "running"
    assert summary.metadata["recovery"]["reason"] == "stale-execution-after-restart"
    assert repository.task_status(task_id) is TaskStatus.INTERRUPTED
    # History is preserved, not rewritten.
    events = await repository.list_events(task_id, latest_run_only=False)
    assert [e.type for e in events] == ["state"]

    # Recovery is idempotent: nothing left open, second pass is a no-op.
    assert await service.recover_stale_executions() == []


async def test_resume_after_recovery_runs_single_attempt(
    repository, settings, broker, orchestrator, background
):
    task_id, _run_id = await _craft_stale_run(
        repository, settings, owner="dead-process"
    )
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=broker,
        settings=settings,
        background=background,
    )
    await service.recover_stale_executions()

    resumed = await service.resume_task(task_id)
    await service.wait_idle()

    assert resumed.id == task_id
    assert len(orchestrator.seen) == 1
    assert await repository.get_latest_run_status(task_id) is RunStatus.COMPLETED
    # The stale attempt plus exactly one fresh attempt, history intact.
    events = await repository.list_events(task_id, latest_run_only=False)
    assert [e.type for e in events] == ["state", "state", "finished"]


async def test_own_live_run_is_not_reaped(
    repository, settings, broker, background
):
    gate = GatedOrchestrator()
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: gate},
        repository=repository,
        broker=broker,
        settings=settings,
        background=background,
    )
    task = await service.create_task(goal="slow")

    for _ in range(200):
        if await repository.get_latest_run_status(task.id) is RunStatus.RUNNING:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("run never reached RUNNING")

    assert await service.recover_stale_executions() == []
    assert await repository.get_latest_run_status(task.id) is RunStatus.RUNNING

    gate.release()
    await service.wait_idle()
    assert await repository.get_latest_run_status(task.id) is RunStatus.COMPLETED


async def test_resume_refuses_foreign_owned_running_run(
    repository, settings, broker, orchestrator, background
):
    task_id, run_id = await _craft_stale_run(
        repository, settings, owner="other-live-process"
    )
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=broker,
        settings=settings,
        background=background,
    )
    with pytest.raises(TaskAlreadyRunning):
        await service.resume_task(task_id)
    # Untouched: still open, no new attempt started.
    assert await repository.get_latest_run_status(task_id) is RunStatus.RUNNING
    assert await repository.get_latest_run_id(task_id) == run_id
    assert orchestrator.seen == []

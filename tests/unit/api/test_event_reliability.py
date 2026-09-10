"""Authoritative events must never fail silently (P0-6).

The repository is the execution history; the broker is delivery. So a broker
hiccup must not fail a run whose history persisted fine, while a persistence
failure must never let a run report success — and a totally unwritable store
leaves the run honestly non-terminal for restart recovery to reap.
"""

from __future__ import annotations

from common.enums import RunStatus


async def test_broker_publish_failure_does_not_fail_run(
    task_service, repository, broker, monkeypatch
) -> None:
    """Delivery is a cache: history persisted, run still completes."""

    async def boom(task_id, event):
        raise RuntimeError("broker down")

    monkeypatch.setattr(broker, "publish", boom)

    task = await task_service.create_task(goal="stream me")
    await task_service.wait_idle()

    assert await repository.get_latest_run_status(task.id) is RunStatus.COMPLETED
    events = await repository.list_events(task.id)
    assert [e.type for e in events] == ["state", "finished"]


async def test_repo_append_failure_never_reports_success(
    task_service, repository, broker, orchestrator, settings, background
) -> None:
    """Persistence failure surfaces as non-terminal, never as success.

    With every event write failing, the run cannot record anything — so it
    must not report COMPLETED. It stays honestly open until restart recovery
    reaps it as INTERRUPTED with its marker.
    """
    from api.services.task_service import TaskService
    from common.enums import AgentTrack

    async def boom(run_id, event, sequence_number):
        raise RuntimeError("disk on fire")

    repository.append_event = boom  # type: ignore[method-assign]

    task = await task_service.create_task(goal="doomed")
    await task_service.wait_idle()

    assert await repository.get_latest_run_status(task.id) is RunStatus.RUNNING

    # A fresh process (fresh owner, empty registries) reaps it — never completed.
    service2 = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=broker,
        settings=settings,
        background=set(),
    )
    recovered = await service2.recover_stale_executions()
    assert len(recovered) == 1
    assert await repository.get_latest_run_status(task.id) is RunStatus.INTERRUPTED
    summary = await repository.get_latest_run(task.id)
    assert summary is not None
    assert summary.metadata["recovery"]["reason"] == "stale-execution-after-restart"

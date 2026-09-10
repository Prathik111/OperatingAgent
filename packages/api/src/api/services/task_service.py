"""``TaskService`` — the orchestration seam between the HTTP layer and the tracks.

It accepts a goal, opens a task + run in the repository, and dispatches the run
to the track's ``IAgentOrchestrator`` on a background task so the ``POST`` can
return ``202`` immediately. As the orchestrator emits events, the service
persists each one (ordered) and republishes it to the broker for SSE/WebSocket
subscribers. When the run ends it records the terminal outcome and closes the
broker topic so every subscriber's stream terminates.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus, TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord
from common.interfaces import IAgentOrchestrator

from ..config import ApiSettings
from ..errors import TaskAlreadyRunning, TaskNotInThread, ThreadNotFound, UnknownTrack
from ..repository.base import RunSummary, TaskRepository, ThreadRecord
from ..workspace import resolve_workspace
from .approval_gateway import ApprovalGateway
from .event_broker import EventBroker

log = logging.getLogger(__name__)

#: Terminal run status -> the coarse task status persisted alongside it.
_RUN_TO_TASK = {
    RunStatus.COMPLETED: TaskStatus.COMPLETED,
    RunStatus.FAILED: TaskStatus.FAILED,
    RunStatus.INTERRUPTED: TaskStatus.INTERRUPTED,
}


def _task_status_for(run_status: RunStatus) -> TaskStatus:
    return _RUN_TO_TASK.get(run_status, TaskStatus.EXECUTING)


class TaskService:
    def __init__(
        self,
        *,
        orchestrators: dict[AgentTrack, IAgentOrchestrator],
        repository: TaskRepository,
        broker: EventBroker,
        settings: ApiSettings,
        background: set[asyncio.Task],
        approvals: ApprovalGateway | None = None,
        execution_owner: str | None = None,
    ) -> None:
        self._orchestrators = orchestrators
        self._repo = repository
        self._broker = broker
        self._approvals = approvals or ApprovalGateway(repository=repository)
        self._settings = settings
        self._active_task_ids: set[str] = set()
        self._active_thread_ids: set[str] = set()
        # Names this process's execution claims in durable run metadata. A
        # fresh token per service means a later process (same code, new owner)
        # can tell live claims — impossible, it just started — from stale ones
        # left behind by a dead process. The in-memory sets above stay the
        # fast path; the database stays the authority.
        self._execution_owner = execution_owner or uuid4().hex
        self._lifecycle_guard = asyncio.Lock()
        self._thread_locks: dict[str, asyncio.Lock] = {}
        # Held so the event loop keeps a strong ref — asyncio only weak-refs
        # tasks, so a fire-and-forget run could otherwise be GC'd mid-flight.
        self._background = background

    @property
    def available_tracks(self) -> list[str]:
        return [t.value for t in self._orchestrators]

    @property
    def execution_owner(self) -> str:
        """This process's execution-claim token (see ``recover_stale_executions``)."""
        return self._execution_owner

    async def recover_stale_executions(self) -> list[str]:
        """Reap runs left non-terminal by a dead process.

        Called once at startup, before serving requests: at that point nothing
        can be live in this process, so every open run whose owner is not ours
        is definitionally stale. Each is closed as INTERRUPTED with a recovery
        marker in its metadata (history and events are untouched), its task
        status follows, and the run id is reported. Runs owned by this process
        are never reaped here — a live owner reaps nothing.

        Returns the recovered run ids. Raises nothing for an empty store.
        """
        recovered: list[str] = []
        for open_run in await self._repo.list_open_runs():
            if open_run.metadata.get("execution_owner") == self._execution_owner:
                continue
            await self._recover_run(
                open_run.task_id,
                open_run.run_id,
                open_run.status,
                dict(open_run.metadata),
                reason="stale-execution-after-restart",
            )
            recovered.append(open_run.run_id)
        if recovered:
            log.warning(
                "recovered %d stale run(s) left non-terminal by a previous process: %s",
                len(recovered),
                recovered,
            )
        return recovered

    async def _recover_run(
        self,
        task_id: str,
        run_id: str,
        previous_status: RunStatus,
        metadata: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        """Close one stale run as INTERRUPTED, preserving its history.

        The run's events, outputs and prior metadata are left exactly as they
        were; only the terminal status, an error line and a ``recovery``
        marker are added, so a later resume starts a clean new attempt with a
        full audit trail of what came before.
        """
        result = AgentRunResult(
            status=RunStatus.INTERRUPTED,
            output=None,
            duration_ms=0.0,
            llm_calls=0,
            tool_calls=0,
            total_tokens=0,
            metadata={
                **metadata,
                "error": (
                    f"execution did not survive (was {previous_status.value}); "
                    "safe to resume for a fresh attempt"
                ),
                "recovery": {
                    "reason": reason,
                    "previous_status": previous_status.value,
                    "recovered_by": self._execution_owner,
                },
            },
        )
        await self._repo.finalize_run(run_id, result)
        await self._repo.update_task_status(task_id, TaskStatus.INTERRUPTED)

    @property
    def orchestrators(self) -> dict[str, IAgentOrchestrator]:
        return {track.value: orchestrator for track, orchestrator in self._orchestrators.items()}

    async def create_thread(self, title: str | None = None) -> ThreadRecord:
        """Create an empty thread so a chat can exist before its first task."""
        return await self._repo.create_thread(str(uuid4()), title)

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and everything under it; False when unknown."""
        async with self._lifecycle_guard:
            lock = self._thread_locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            if thread_id in self._active_thread_ids:
                raise TaskAlreadyRunning(thread_id)
            deleted = await self._repo.delete_thread(thread_id)
            if deleted:
                self._active_thread_ids.discard(thread_id)
            return deleted

    async def create_task(
        self,
        goal: str,
        track: AgentTrack | None = None,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        workspace: str | None = None,
    ) -> AgentTask:
        resolved_track = track or self._settings.default_track
        if resolved_track not in self._orchestrators:
            raise UnknownTrack(str(resolved_track))

        resolved_thread_id = thread_id or str(uuid4())
        async with self._lifecycle_guard:
            lifecycle_lock = self._thread_locks.setdefault(
                resolved_thread_id, asyncio.Lock()
            )
        await lifecycle_lock.acquire()
        if resolved_thread_id in self._active_thread_ids:
            lifecycle_lock.release()
            raise TaskAlreadyRunning(resolved_thread_id)
        self._active_thread_ids.add(resolved_thread_id)
        try:
            continuing_thread = False
            existing_tasks: list[tuple[AgentTask, RunStatus | None]] = []
            if thread_id is not None:
                try:
                    existing_tasks = await self._repo.list_tasks_by_thread(
                        thread_id, limit=1, offset=0
                    )
                    continuing_thread = bool(existing_tasks)
                except ThreadNotFound:
                    # ``save_task`` creates a new thread in the Postgres backend;
                    # an unknown explicit id therefore represents its first turn.
                    continuing_thread = False

            task_metadata = dict(metadata or {})
            selected_workspace = workspace or task_metadata.get("workspace") or task_metadata.get(
                "working_directory"
            )
            if selected_workspace is None and existing_tasks:
                previous = existing_tasks[0][0].metadata
                selected_workspace = previous.get("workspace") or previous.get(
                    "working_directory"
                )
            resolved_workspace = resolve_workspace(
                str(selected_workspace) if selected_workspace else None,
                default=self._settings.sandbox_workspace,
            )
            task_metadata["workspace"] = resolved_workspace
            # Native's internal Session still calls this working_directory.
            task_metadata["working_directory"] = resolved_workspace

            task = AgentTask(
                id=str(uuid4()),
                goal=goal,
                thread_id=resolved_thread_id,
                track=resolved_track,
                metadata=task_metadata,
                execution_mode="continue" if continuing_thread else "new",
            )
            await self._repo.save_task(task)
            config = self._settings.build_agent_config(resolved_track)
            run_id = await self._repo.create_run(
                task.id,
                config,
                metadata={
                    "execution_mode": task.execution_mode,
                    "thread_id": task.thread_id,
                    "checkpoint_namespace": config.checkpoint.namespace,
                    "execution_owner": self._execution_owner,
                },
            )

            await self._broker.reopen(task.id, clear=True)
            self._active_task_ids.add(task.id)
            run_task = asyncio.create_task(self._run(task, run_id))
            self._background.add(run_task)
            run_task.add_done_callback(self._background.discard)
            return task
        except BaseException:
            self._active_thread_ids.discard(resolved_thread_id)
            raise
        finally:
            lifecycle_lock.release()

    async def create_thread_task(
        self,
        thread_id: str,
        goal: str,
        track: AgentTrack | None = None,
        metadata: dict[str, Any] | None = None,
        workspace: str | None = None,
    ) -> AgentTask:
        """Create a new turn in an existing thread after ownership validation."""
        existing = await self._repo.list_tasks_by_thread(thread_id, limit=1, offset=0)
        if workspace is None and existing:
            previous = existing[0][0].metadata
            workspace = str(
                previous.get("workspace") or previous.get("working_directory") or ""
            ) or None
        return await self.create_task(
            goal=goal,
            track=track,
            thread_id=thread_id,
            metadata=metadata,
            workspace=workspace,
        )

    async def resume_task(
        self,
        task_id: str,
        *,
        resume_value: object | None = None,
        checkpoint_id: str | None = None,
    ) -> AgentTask:
        """Start another attempt from the latest LangGraph checkpoint."""
        async with self._lifecycle_guard:
            task = await self._repo.get_task(task_id)
            lifecycle_lock = self._thread_locks.setdefault(
                task.thread_id, asyncio.Lock()
            )
        async with lifecycle_lock:
            if task.thread_id in self._active_thread_ids:
                raise TaskAlreadyRunning(task.thread_id)
            self._active_thread_ids.add(task.thread_id)
            try:
                latest_status = await self._repo.get_latest_run_status(task_id)
                if latest_status in {
                    RunStatus.CREATED,
                    RunStatus.PENDING,
                } or (
                    latest_status is RunStatus.RUNNING
                    and task_id in self._active_task_ids
                ):
                    raise TaskAlreadyRunning(task_id)
                if (
                    latest_status is RunStatus.RUNNING
                    and task_id not in self._active_task_ids
                ):
                    # RUNNING in the database but with no live execution in this
                    # process: a stale claim (typically a dead process that never
                    # got reaped). Close it as recovered before opening a new
                    # attempt, so history never shows two open executions. A run
                    # owned by a *different* live owner is refused instead — it
                    # may genuinely still be executing elsewhere.
                    latest_metadata = await self._repo.get_latest_run_metadata(task_id)
                    owner = latest_metadata.get("execution_owner")
                    if owner is not None and owner != self._execution_owner:
                        raise TaskAlreadyRunning(task_id)
                    latest_run_id = await self._repo.get_latest_run_id(task_id)
                    if latest_run_id is not None:
                        log.warning(
                            "resuming over stale run %s (status %s); closing it as recovered first",
                            latest_run_id,
                            latest_status.value,
                        )
                        await self._recover_run(
                            task_id,
                            latest_run_id,
                            latest_status,
                            dict(latest_metadata),
                            reason="stale-execution-at-resume",
                        )

                previous_run_id = await self._repo.get_latest_run_id(task_id)
                previous_metadata = await self._repo.get_latest_run_metadata(task_id)
                task.execution_mode = "resume"
                task.resume_value = resume_value
                task.resume_checkpoint_id = checkpoint_id
                task.resume_checkpoint_namespace = previous_metadata.get(
                    "checkpoint_namespace"
                )
                config = self._settings.build_agent_config(task.track)
                run_id = await self._repo.create_run(
                    task.id,
                    config,
                    metadata={
                        "execution_mode": "resume",
                        "thread_id": task.thread_id,
                        "checkpoint_namespace": (
                            task.resume_checkpoint_namespace
                            or config.checkpoint.namespace
                        ),
                        "resumes_run_id": previous_run_id,
                        "checkpoint_id": checkpoint_id,
                        "execution_owner": self._execution_owner,
                    },
                )
                await self._broker.reopen(task.id, clear=True)
                self._active_task_ids.add(task.id)
                run_task = asyncio.create_task(self._run(task, run_id))
                self._background.add(run_task)
                run_task.add_done_callback(self._background.discard)
                return task
            except BaseException:
                self._active_thread_ids.discard(task.thread_id)
                raise

    async def resume_task_in_thread(
        self,
        thread_id: str,
        task_id: str,
        *,
        resume_value: object | None = None,
        checkpoint_id: str | None = None,
    ) -> AgentTask:
        await self.get_task_in_thread(thread_id, task_id)
        return await self.resume_task(
            task_id,
            resume_value=resume_value,
            checkpoint_id=checkpoint_id,
        )

    async def get_task(self, task_id: str) -> tuple[AgentTask, RunStatus | None]:
        task = await self._repo.get_task(task_id)  # raises TaskNotFound
        status = await self._repo.get_latest_run_status(task_id)
        return task, status

    async def get_task_details(
        self, task_id: str
    ) -> tuple[AgentTask, RunSummary | None]:
        task = await self._repo.get_task(task_id)
        return task, await self._repo.get_latest_run(task_id)

    async def get_task_in_thread(
        self, thread_id: str, task_id: str
    ) -> tuple[AgentTask, RunSummary | None]:
        task, run = await self.get_task_details(task_id)
        if task.thread_id != thread_id:
            raise TaskNotInThread(task_id, thread_id)
        return task, run

    async def list_threads(self, *, limit: int, offset: int) -> list[ThreadRecord]:
        return await self._repo.list_threads(limit=limit, offset=offset)

    async def list_thread_tasks(
        self, thread_id: str, *, limit: int, offset: int
    ) -> list[tuple[AgentTask, RunStatus | None]]:
        return await self._repo.list_tasks_by_thread(
            thread_id,
            limit=limit,
            offset=offset,
        )

    async def stream_task(self, task_id: str) -> AsyncIterator[AgentEvent]:
        """Hydrate persisted events, then yield the live/replay event stream."""
        await self._repo.get_task(task_id)
        events = await self._repo.list_events(task_id)
        status = await self._repo.get_latest_run_status(task_id)
        await self._broker.hydrate(
            task_id,
            events,
            closed=status in _RUN_TO_TASK,
        )
        async for event in self._broker.subscribe(task_id):
            yield event

    async def list_thread_task_details(
        self, thread_id: str, *, limit: int, offset: int
    ) -> list[tuple[AgentTask, RunSummary | None]]:
        tasks = await self._repo.list_tasks_by_thread(
            thread_id, limit=limit, offset=offset
        )
        return [
            (task, await self._repo.get_latest_run(task.id)) for task, _status in tasks
        ]

    async def list_thread_events(self, thread_id: str) -> list[tuple[str, AgentEvent]]:
        await self._repo.list_tasks_by_thread(thread_id, limit=1, offset=0)
        return await self._repo.list_thread_events(thread_id)

    async def wait_idle(self) -> None:
        """Await all in-flight background runs — for tests and graceful drain."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    # -- background run ----------------------------------------------------

    async def _run(self, task: AgentTask, run_id: str) -> None:
        sequence = itertools.count()

        # A resumed attempt can reuse successful side-effecting tool results
        # recorded by an earlier attempt. This is intentionally event-based so
        # it works with both repository backends without duplicating tool data.
        try:
            history = await self._repo.list_events(task.id, latest_run_only=False)
        except Exception as exc:  # noqa: BLE001 - recovery hint is best effort
            log.warning("could not load prior tool events for %s: %s", task.id, exc)
            history = []
        task.completed_tool_calls = {
            str(event.payload["call_id"]): str(event.payload.get("output", ""))
            for event in history
            if event.type == "tool_finished"
            and event.payload.get("success") is True
            and event.payload.get("call_id")
        }

        async def on_event(event: AgentEvent) -> None:
            # Persist first (ordered, durable), then fan out to subscribers.
            # Persistence failures propagate: the orchestrator records the run
            # FAILED rather than reporting success with a holey history.
            # Delivery alone is best-effort — the broker is a cache, so a
            # publish hiccup is logged loudly but must not fail a run whose
            # history persisted fine.
            await self._repo.append_event(run_id, event, next(sequence))
            if event.type == "llm_call":
                await self._repo.save_llm_call(
                    run_id, LLMCallRecord.from_payload(event.payload)
                )
            elif event.type == "tool_call":
                await self._repo.save_tool_call(
                    run_id, ToolCallRecord.from_payload(event.payload)
                )
            elif event.type == "phase_entered":
                await self._repo.save_phase(run_id, event.payload)
            elif event.type == "phase_exited":
                await self._repo.close_phase(run_id, event.payload)
            elif event.type == "plan_created":
                await self._repo.save_plan(run_id, event.payload)
            elif event.type == "finding_recorded":
                await self._repo.save_finding(run_id, event.payload)
            elif event.type == "verification_recorded":
                await self._repo.save_verification(run_id, event.payload)
            elif event.type == "trace_ref":
                await self._repo.save_trace_ref(run_id, event.payload)
            elif event.type == "approval_requested":
                await self._repo.save_approval(run_id, event.payload)
            elif event.type == "approval_resolved":
                await self._repo.resolve_approval(event.payload)
            try:
                await self._broker.publish(task.id, event)
            except Exception as exc:  # noqa: BLE001 - delivery is a cache, not history
                log.warning(
                    "event delivery failed for task %s (history persisted): %s",
                    task.id,
                    exc,
                )

        try:
            await self._repo.mark_run_running(run_id)
            orchestrator = self._orchestrators[task.track]
            try:
                result = await orchestrator.run(task, on_event=on_event)
            except asyncio.CancelledError:
                # Shutdown/cancel: shield the terminal write so the run isn't
                # left dangling as 'running', then propagate the cancellation.
                await asyncio.shield(self._finalize_cancelled(task, run_id))
                raise
            except Exception as exc:  # an orchestrator that broke its contract
                log.exception("orchestrator raised for task %s", task.id)
                await on_event(AgentEvent(type="error", payload={"error": str(exc)}))
                result = AgentRunResult(
                    status=RunStatus.FAILED,
                    output=None,
                    duration_ms=0.0,
                    llm_calls=0,
                    tool_calls=0,
                    total_tokens=0,
                    metadata={"error": str(exc)},
                )

            await self._repo.finalize_run(run_id, result)
            await self._repo.update_task_status(
                task.id, _task_status_for(result.status)
            )
        finally:
            # Terminal sentinel: drains every SSE/WebSocket subscriber.
            await self._broker.close(task.id)
            self._active_task_ids.discard(task.id)
            self._active_thread_ids.discard(task.thread_id)

    async def _finalize_cancelled(self, task: AgentTask, run_id: str) -> None:
        result = AgentRunResult(
            status=RunStatus.INTERRUPTED,
            output=None,
            duration_ms=0.0,
            llm_calls=0,
            tool_calls=0,
            total_tokens=0,
            metadata={"error": "run cancelled"},
        )
        await self._repo.finalize_run(run_id, result)
        await self._repo.update_task_status(task.id, TaskStatus.INTERRUPTED)

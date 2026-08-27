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
from uuid import uuid4

from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus, TaskStatus
from common.events import AgentEvent
from common.interfaces import IAgentOrchestrator

from packages.api.src.api.repository.base import TaskRepository

from ..config import ApiSettings
from ..errors import UnknownTrack
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
        approvals: ApprovalGateway,
        settings: ApiSettings,
        background: set[asyncio.Task],
    ) -> None:
        self._orchestrators = orchestrators
        self._repo = repository
        self._broker = broker
        self._approvals = approvals
        self._settings = settings
        # Held so the event loop keeps a strong ref — asyncio only weak-refs
        # tasks, so a fire-and-forget run could otherwise be GC'd mid-flight.
        self._background = background

    @property
    def available_tracks(self) -> list[str]:
        return [t.value for t in self._orchestrators]

    async def create_task(
        self,
        goal: str,
        track: AgentTrack | None = None,
        thread_id: str | None = None,
        metadata: dict | None = None,
    ) -> AgentTask:
        resolved_track = track or self._settings.default_track
        if resolved_track not in self._orchestrators:
            raise UnknownTrack(str(resolved_track))

        task = AgentTask(
            id=str(uuid4()),
            goal=goal,
            thread_id=thread_id or str(uuid4()),
            track=resolved_track,
            metadata=metadata or {},
        )
        await self._repo.save_task(task)
        config = self._settings.build_agent_config(resolved_track)
        run_id = await self._repo.create_run(task.id, config)

        run_task = asyncio.create_task(self._run(task, run_id))
        self._background.add(run_task)
        run_task.add_done_callback(self._background.discard)
        return task

    async def get_task(self, task_id: str) -> tuple[AgentTask, RunStatus | None]:
        task = await self._repo.get_task(task_id)  # raises TaskNotFound
        status = await self._repo.get_latest_run_status(task_id)
        return task, status

    def stream_task(self, task_id: str):
        """Return an async iterator of the task's events (buffered + live)."""
        return self._broker.subscribe(task_id)

    async def wait_idle(self) -> None:
        """Await all in-flight background runs — for tests and graceful drain."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    # -- background run ----------------------------------------------------

    async def _run(self, task: AgentTask, run_id: str) -> None:
        sequence = itertools.count()

        async def on_event(event: AgentEvent) -> None:
            # Persist first (ordered, durable), then fan out to subscribers.
            await self._repo.append_event(run_id, event, next(sequence))
            await self._broker.publish(task.id, event)

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

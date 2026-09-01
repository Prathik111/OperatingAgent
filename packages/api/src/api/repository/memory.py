"""In-memory ``TaskRepository`` — the default, fully hermetic backend.

Plain dicts, no I/O. It is the store the unit suite runs against and the
sensible default for local dev where a Postgres instance is overkill. It keeps
the same task/run/event shape as the Postgres store so switching backends
changes nothing above the repository seam.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from common.agent import AgentRunResult, AgentTask
from common.config import AgentConfig
from common.enums import RunStatus, TaskStatus
from common.events import AgentEvent

from ..errors import TaskNotFound, ThreadNotFound
from .base import ThreadRecord


@dataclass(slots=True)
class _Run:
    id: str
    task_id: str
    status: RunStatus
    order: int
    output: str | None = None
    last_error: str | None = None
    events: list[tuple[int, str, dict]] = field(default_factory=list)


@dataclass(slots=True)
class _Thread:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class InMemoryTaskRepository:
    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._threads: dict[str, _Thread] = {}
        self._task_status: dict[str, TaskStatus] = {}
        self._runs: dict[str, _Run] = {}
        self._order = itertools.count()

    async def save_task(self, task: AgentTask) -> None:
        self._tasks[task.id] = task
        thread = self._threads.get(task.thread_id)
        if thread is None:
            self._threads[task.thread_id] = _Thread(
                id=task.thread_id,
                title=task.metadata.get("title"),
                created_at=task.created_at,
                updated_at=task.created_at,
            )
        else:
            thread.updated_at = max(thread.updated_at, task.created_at)
        self._task_status.setdefault(task.id, TaskStatus.PLANNING)

    async def get_task(self, task_id: str) -> AgentTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFound(task_id) from None

    async def list_threads(self, *, limit: int, offset: int) -> list[ThreadRecord]:
        threads = sorted(
            self._threads.values(),
            key=lambda thread: (thread.updated_at, thread.id),
            reverse=True,
        )
        selected = threads[offset : offset + limit]
        return [
            ThreadRecord(
                id=thread.id,
                title=thread.title,
                task_count=sum(
                    task.thread_id == thread.id for task in self._tasks.values()
                ),
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
            for thread in selected
        ]

    async def list_tasks_by_thread(
        self, thread_id: str, *, limit: int, offset: int
    ) -> list[tuple[AgentTask, RunStatus | None]]:
        if thread_id not in self._threads:
            raise ThreadNotFound(thread_id)
        tasks = sorted(
            (task for task in self._tasks.values() if task.thread_id == thread_id),
            key=lambda task: (task.created_at, task.id),
            reverse=True,
        )
        return [
            (task, self._latest_run_status(task.id))
            for task in tasks[offset : offset + limit]
        ]

    async def create_run(self, task_id: str, config: AgentConfig) -> str:
        run_id = str(uuid4())
        self._runs[run_id] = _Run(
            id=run_id,
            task_id=task_id,
            status=RunStatus.CREATED,
            order=next(self._order),
        )
        return run_id

    async def mark_run_running(self, run_id: str) -> None:
        self._runs[run_id].status = RunStatus.RUNNING

    async def append_event(
        self, run_id: str, event: AgentEvent, sequence_number: int
    ) -> None:
        self._runs[run_id].events.append((sequence_number, event.type, event.payload))

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        run = self._runs[run_id]
        run.status = result.status
        run.output = result.output
        run.last_error = result.metadata.get("error")

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        self._task_status[task_id] = status

    async def get_latest_run_status(self, task_id: str) -> RunStatus | None:
        return self._latest_run_status(task_id)

    def _latest_run_status(self, task_id: str) -> RunStatus | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        return max(runs, key=lambda r: r.order).status

    # -- test/introspection helpers (not part of the Protocol) --------------

    def events_for(self, run_id: str) -> list[tuple[int, str, dict]]:
        return list(self._runs[run_id].events)

    def task_status(self, task_id: str) -> TaskStatus | None:
        return self._task_status.get(task_id)

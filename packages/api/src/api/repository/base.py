"""The persistence contract the service layer depends on.

``TaskRepository`` is a structural ``Protocol`` so the service never learns
which backend it holds — the in-memory store (default, hermetic) and the
Postgres store (the system of record) both satisfy it. The method set maps onto
the class diagram's ``save`` / ``get`` / ``log_metrics`` while modelling the run
lifecycle the schema actually records: a task, a run under it, ordered events
on the run, and the run's terminal outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from common.agent import AgentRunResult, AgentTask
from common.config import AgentConfig
from common.enums import RunStatus, TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord


@dataclass(slots=True, frozen=True)
class ThreadRecord:
    """Repository-neutral summary of one conversation thread."""

    id: str
    title: str | None
    task_count: int
    created_at: datetime
    updated_at: datetime


@runtime_checkable
class TaskRepository(Protocol):
    async def save_task(self, task: AgentTask) -> None:
        """Persist a task (and, for Postgres, its owning actor + thread)."""
        ...

    async def get_task(self, task_id: str) -> AgentTask:
        """Return a task or raise ``TaskNotFound``."""
        ...

    async def list_threads(self, *, limit: int, offset: int) -> list[ThreadRecord]:
        """Return API-owned threads ordered by most recent activity."""
        ...

    async def list_tasks_by_thread(
        self, thread_id: str, *, limit: int, offset: int
    ) -> list[tuple[AgentTask, RunStatus | None]]:
        """Return a thread's tasks and latest run statuses or raise ThreadNotFound."""
        ...

    async def create_run(self, task_id: str, config: AgentConfig) -> str:
        """Open a run for a task against a config snapshot; return its run id."""
        ...

    async def mark_run_running(self, run_id: str) -> None:
        """Move a run to ``running`` and stamp its start time."""
        ...

    async def append_event(
        self, run_id: str, event: AgentEvent, sequence_number: int
    ) -> None:
        """Append one ordered event to a run."""
        ...

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        """Record a run's terminal status, output and error."""
        ...

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update the task's coarse status."""
        ...

    async def get_latest_run_status(self, task_id: str) -> RunStatus | None:
        """The status of the task's most recent run, or ``None`` if none exists."""
        ...

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None: ...

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None: ...

    async def save_phase(self, run_id: str, payload: dict) -> str: ...

    async def close_phase(self, run_id: str, payload: dict) -> None: ...

    async def save_plan(self, run_id: str, payload: dict) -> str: ...

    async def save_finding(self, run_id: str, payload: dict) -> str: ...

    async def save_verification(self, run_id: str, payload: dict) -> str: ...

    async def save_trace_ref(self, run_id: str, payload: dict) -> str: ...

    async def save_approval(self, run_id: str, payload: dict) -> str: ...

    async def resolve_approval(self, payload: dict) -> None: ...

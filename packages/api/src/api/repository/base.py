"""The persistence contract the service layer depends on.

``TaskRepository`` is a structural ``Protocol`` so the service never learns
which backend it holds — the in-memory store (default, hermetic) and the
Postgres store (the system of record) both satisfy it. The method set maps onto
the class diagram's ``save`` / ``get`` / ``log_metrics`` while modelling the run
lifecycle the schema actually records: a task, a run under it, ordered events
on the run, and the run's terminal outcome.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from common.agent import AgentRunResult, AgentTask
from common.config import AgentConfig
from common.enums import RunStatus, TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord


@runtime_checkable
class TaskRepository(Protocol):
    async def save_task(self, task: AgentTask) -> None:
        """Persist a task (and, for Postgres, its owning actor + thread)."""
        ...

    async def get_task(self, task_id: str) -> AgentTask:
        """Return a task or raise ``TaskNotFound``."""
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

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        """Persist one model invocation for run metrics and reconciliation."""
        ...

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        """Persist one resolved tool invocation."""
        ...

    async def upsert_tool(
        self, server_name: str, base_url: str | None, tool_spec: dict
    ) -> str:
        """Register a discovered tool and return its canonical id."""
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

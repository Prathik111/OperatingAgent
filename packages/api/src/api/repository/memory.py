"""In-memory ``TaskRepository`` — the default, fully hermetic backend.

Plain dicts, no I/O. It is the store the unit suite runs against and the
sensible default for local dev where a Postgres instance is overkill. It keeps
the same task/run/event shape as the Postgres store so switching backends
changes nothing above the repository seam.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from uuid import uuid4

from common.agent import AgentRunResult, AgentTask
from common.config import AgentConfig
from common.enums import RunStatus, TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord

from ..errors import TaskNotFound


@dataclass(slots=True)
class _Run:
    id: str
    task_id: str
    status: RunStatus
    order: int
    output: str | None = None
    last_error: str | None = None
    events: list[tuple[int, str, dict]] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    trace_refs: list[dict] = field(default_factory=list)
    approvals: list[dict] = field(default_factory=list)


class InMemoryTaskRepository:
    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._task_status: dict[str, TaskStatus] = {}
        self._runs: dict[str, _Run] = {}
        self._order = itertools.count()
        self._tools: dict[tuple[str, str], tuple[str, dict]] = {}

    async def save_task(self, task: AgentTask) -> None:
        self._tasks[task.id] = task
        self._task_status.setdefault(task.id, TaskStatus.PLANNING)

    async def get_task(self, task_id: str) -> AgentTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFound(task_id) from None

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

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        self._runs[run_id].llm_calls.append(record)

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        tool_id = await self.upsert_tool(
            record.server_name,
            record.base_url,
            {
                "name": record.tool_name,
                "description": record.description,
                "input_schema": record.input_schema,
            },
        )
        from dataclasses import replace
        self._runs[run_id].tool_calls.append(replace(record, tool_id=tool_id))

    async def save_phase(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].phases.append(value)
        return value["id"]

    async def close_phase(self, run_id: str, payload: dict) -> None:
        for phase in self._runs[run_id].phases:
            if phase["id"] == payload["phase_id"]:
                phase.update(payload)
                return

    async def save_plan(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].plans.append(value)
        return value["id"]

    async def save_finding(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].findings.append(value)
        return value["id"]

    async def save_verification(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].verifications.append(value)
        return value["id"]

    async def save_trace_ref(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].trace_refs.append(value)
        return value["id"]

    async def save_approval(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4()), "status": "pending"}
        self._runs[run_id].approvals.append(value)
        return value["id"]

    async def resolve_approval(self, payload: dict) -> None:
        for run in self._runs.values():
            for approval in run.approvals:
                if approval["id"] == payload["approval_id"]:
                    approval.update(payload)
                    approval["status"] = "approved" if payload["approved"] else "denied"
                    return

    async def upsert_tool(
        self, server_name: str, base_url: str | None, tool_spec: dict
    ) -> str:
        key = (server_name, str(tool_spec["name"]))
        existing = self._tools.get(key)
        tool_id = existing[0] if existing else str(uuid4())
        self._tools[key] = (tool_id, {**tool_spec, "base_url": base_url})
        return tool_id

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        run = self._runs[run_id]
        run.status = result.status
        run.output = result.output
        run.last_error = result.metadata.get("error")

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        self._task_status[task_id] = status

    async def get_latest_run_status(self, task_id: str) -> RunStatus | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        return max(runs, key=lambda r: r.order).status

    # -- test/introspection helpers (not part of the Protocol) --------------

    def events_for(self, run_id: str) -> list[tuple[int, str, dict]]:
        return list(self._runs[run_id].events)

    def llm_calls_for(self, run_id: str) -> list[LLMCallRecord]:
        return list(self._runs[run_id].llm_calls)

    def tool_calls_for(self, run_id: str) -> list[ToolCallRecord]:
        return list(self._runs[run_id].tool_calls)

    def task_status(self, task_id: str) -> TaskStatus | None:
        return self._task_status.get(task_id)

"""agent-native's own persistence layer (decision #9: Postgres backend).

This is a from-scratch repository - it deliberately does not share types or
code with any other package. Backends:
  - PostgresTaskRepository: asyncpg, schema bootstrapped on first connect
    (v1 simplification, documented). DSN from config.database_url /
    AGENT_NATIVE_DATABASE_URL.
  - InMemoryTaskRepository: dict-backed; used by tests, the CLI fallback, and
    anywhere Postgres is unreachable. Same interface, so swapping is a wiring
    decision, not a code change.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import asyncpg

from ..types import (
    AgentRunResult,
    AgentTask,
    Plan,
    PlanStep,
    RunStatus,
    StepKind,
    StepStatus,
)


class TaskRepository(Protocol):
    async def save_task(self, task: AgentTask) -> None: ...

    async def get_task(self, task_id: str) -> AgentTask | None: ...

    async def save_plan(self, plan: Plan) -> None: ...

    async def get_plan(self, task_id: str) -> Plan | None: ...

    async def save_run_result(self, result: AgentRunResult) -> None: ...

    async def list_run_results(self, task_id: str) -> list[AgentRunResult]: ...

    async def close(self) -> None: ...


def _step_to_row(plan_id: str, step: PlanStep) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "step_id": step.id,
        "position": None,
        "description": step.description,
        "kind": step.kind.value,
        "tool_name": step.tool_name,
        "check": step.check,
        "status": step.status.value,
    }


class InMemoryTaskRepository:
    """Dict-backed TaskRepository (tests / CLI fallback)."""

    def __init__(self) -> None:
        self.tasks: dict[str, AgentTask] = {}
        self.plans: dict[str, Plan] = {}
        self.results: dict[str, list[AgentRunResult]] = {}

    async def save_task(self, task: AgentTask) -> None:
        self.tasks[task.id] = task

    async def get_task(self, task_id: str) -> AgentTask | None:
        return self.tasks.get(task_id)

    async def save_plan(self, plan: Plan) -> None:
        self.plans[plan.task_id] = plan

    async def get_plan(self, task_id: str) -> Plan | None:
        return self.plans.get(task_id)

    async def save_run_result(self, result: AgentRunResult) -> None:
        self.results.setdefault(result.metadata.get("task_id", ""), []).append(result)

    async def list_run_results(self, task_id: str) -> list[AgentRunResult]:
        return self.results.get(task_id, [])

    async def close(self) -> None:
        pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_native_tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    thread_id TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS agent_native_plans (
    task_id TEXT PRIMARY KEY REFERENCES agent_native_tasks(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    steps JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_native_runs (
    id BIGSERIAL PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES agent_native_tasks(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    output TEXT,
    duration_ms DOUBLE PRECISION NOT NULL,
    llm_calls INT NOT NULL,
    tool_calls INT NOT NULL,
    total_tokens INT NOT NULL,
    cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    replans INT NOT NULL DEFAULT 0,
    failure_reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PostgresTaskRepository:
    """asyncpg-backed TaskRepository. Bootstraps its own schema on connect."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def _connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4)
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)

    async def save_task(self, task: AgentTask) -> None:
        await self._connect()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO agent_native_tasks (id, goal, thread_id, metadata, created_at) "
                "VALUES ($1, $2, $3, $4::jsonb, $5) ON CONFLICT (id) DO UPDATE SET "
                "goal = EXCLUDED.goal, metadata = EXCLUDED.metadata",
                task.id, task.goal, task.thread_id, json.dumps(task.metadata), task.created_at,
            )

    async def get_task(self, task_id: str) -> AgentTask | None:
        await self._connect()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, goal, thread_id, metadata, created_at FROM agent_native_tasks WHERE id = $1",
                task_id,
            )
        if row is None:
            return None
        return AgentTask(
            id=row["id"], goal=row["goal"], thread_id=row["thread_id"],
            created_at=row["created_at"], metadata=_decode_json(row["metadata"]),
        )

    async def save_plan(self, plan: Plan) -> None:
        await self._connect()
        steps = [
            {
                "id": s.id, "description": s.description, "kind": s.kind.value,
                "tool_name": s.tool_name, "check": s.check, "status": s.status.value,
                "result": _serialize_result(s.result),
            }
            for s in plan.steps
        ]
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO agent_native_plans (task_id, created_at, steps) VALUES ($1, $2, $3::jsonb) "
                "ON CONFLICT (task_id) DO UPDATE SET steps = EXCLUDED.steps, created_at = EXCLUDED.created_at",
                plan.task_id, plan.created_at, json.dumps(steps),
            )

    async def get_plan(self, task_id: str) -> Plan | None:
        await self._connect()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT steps FROM agent_native_plans WHERE task_id = $1", task_id
            )
        if row is None or not row["steps"]:
            return None
        steps: list[PlanStep] = []
        raw_steps = row["steps"] if isinstance(row["steps"], list) else json.loads(row["steps"])
        for raw in raw_steps:
            steps.append(PlanStep(
                id=raw.get("id", ""),
                description=raw.get("description", ""),
                kind=StepKind(raw.get("kind", StepKind.TOOL.value)),
                tool_name=raw.get("tool_name"),
                check=raw.get("check"),
                status=StepStatus(raw.get("status", StepStatus.PENDING.value)),
                result=_deserialize_result(raw.get("result")),
            ))
        return Plan(task_id=task_id, steps=steps)

    async def save_run_result(self, result: AgentRunResult) -> None:
        await self._connect()
        task_id = result.metadata.get("task_id", "")
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO agent_native_runs (task_id, status, output, duration_ms, llm_calls, "
                "tool_calls, total_tokens, cost, replans, failure_reason, metadata) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)",
                task_id, result.status.value, result.output, result.duration_ms,
                result.llm_calls, result.tool_calls, result.total_tokens,
                result.cost, result.replans, result.failure_reason, json.dumps(result.metadata),
            )

    async def list_run_results(self, task_id: str) -> list[AgentRunResult]:
        await self._connect()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status, output, duration_ms, llm_calls, tool_calls, total_tokens, cost, "
                "replans, failure_reason, metadata FROM agent_native_runs WHERE task_id = $1 "
                "ORDER BY finished_at DESC",
                task_id,
            )
        return [
            AgentRunResult(
                status=RunStatus(r["status"]), output=r["output"],
                duration_ms=r["duration_ms"], llm_calls=r["llm_calls"],
                tool_calls=r["tool_calls"], total_tokens=r["total_tokens"],
                cost=r["cost"], replans=r["replans"], failure_reason=r["failure_reason"],
                metadata=_decode_json(r["metadata"]),
            )
            for r in rows
        ]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _serialize_result(result: Any) -> Any:
    if result is None:
        return None
    return {
        "success": result.success,
        "output": result.output if not isinstance(result.output, (dict, list)) else json.dumps(result.output),
        "error": result.error,
    }


def _deserialize_result(raw: Any) -> Any:
    if raw is None or "success" not in raw:
        return None
    from ..types import ToolCallResult

    return ToolCallResult(success=bool(raw["success"]), output=raw.get("output"), error=raw.get("error"))


def _decode_json(raw: Any) -> dict:
    """asyncpg returns JSONB as str by default unless a codec is registered."""
    import json as _json

    if raw is None or isinstance(raw, dict):
        return dict(raw or {})
    if isinstance(raw, str):
        value = _json.loads(raw)
        return dict(value or {})
    return dict(raw)

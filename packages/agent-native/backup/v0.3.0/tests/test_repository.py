"""TaskRepository tests - in-memory always, Postgres when reachable (#9)."""

from __future__ import annotations

import pytest

from agent_native.repository import (
    InMemoryTaskRepository,
    PostgresTaskRepository,
)
from agent_native.types import AgentRunResult, AgentTask, Plan, PlanStep, RunStatus, StepKind, ToolCallResult
from conftest import make_tool  # noqa: F401  (re-exported convenience)


async def _save_smoke(repo) -> None:
    task = AgentTask(id="task-1", goal="write a report", thread_id="thr-1")
    await repo.save_task(task)
    assert await repo.get_task("task-1") == task
    assert await repo.get_task("missing") is None

    step = PlanStep(id="s1", description="write file", kind=StepKind.TOOL,
                    tool_name="write_file", check="file_exists=report.md")
    plan = Plan(task_id="task-1", steps=[step])
    await repo.save_plan(plan)
    fetched = await repo.get_plan("task-1")
    assert fetched is not None
    assert fetched.steps[0].tool_name == "write_file"
    assert fetched.steps[0].check == "file_exists=report.md"

    result = AgentRunResult(status=RunStatus.COMPLETED, output="done", duration_ms=12.5,
                            llm_calls=3, tool_calls=2, total_tokens=400, replans=1,
                            metadata={"task_id": "task-1"})
    await repo.save_run_result(result)
    results = await repo.list_run_results("task-1")
    assert len(results) == 1
    assert results[0].status == RunStatus.COMPLETED
    assert results[0].replans == 1


@pytest.mark.asyncio
async def test_in_memory_repository_roundtrip():
    repo = InMemoryTaskRepository()
    await _save_smoke(repo)


@pytest.mark.asyncio
async def test_in_memory_plan_update_overwrites():
    repo = InMemoryTaskRepository()
    await repo.save_task(AgentTask(id="t", goal="g"))
    await repo.save_plan(Plan(task_id="t", steps=[PlanStep(id="a", description="1", kind=StepKind.TOOL)]))
    await repo.save_plan(Plan(task_id="t", steps=[PlanStep(id="b", description="2", kind=StepKind.ANALYSIS)]))
    plan = await repo.get_plan("t")
    assert plan is not None and len(plan.steps) == 1 and plan.steps[0].id == "b"


def _postgres_reachable() -> bool:
    import asyncio

    async def _try() -> bool:
        repo = PostgresTaskRepository(
            "postgresql://agent_native:agent_native@localhost:5432/agent_native"
        )
        try:
            await repo._connect()
            return repo._pool is not None
        except Exception:
            return False
        finally:
            await repo.close()

    return asyncio.run(_try())


@pytest.mark.asyncio
@pytest.mark.skipif(not _postgres_reachable(), reason="local Postgres not running (uv run docker compose -f infra/docker/docker-compose.yml up -d)")
async def test_postgres_repository_roundtrip():
    repo = PostgresTaskRepository("postgresql://agent_native:agent_native@localhost:5432/agent_native")
    try:
        await repo._connect()
        async with repo._pool.acquire() as conn:  # type: ignore[union-attr]
            await conn.execute("TRUNCATE agent_native_runs, agent_native_plans, agent_native_tasks")
        await _save_smoke(repo)
    finally:
        await repo.close()
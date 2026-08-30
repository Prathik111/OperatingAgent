from __future__ import annotations

import os
import asyncio
import sys
from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from common.agent import AgentTask
from common.enums import AgentTrack
from packages.api.src.api.config import ApiSettings
from packages.api.src.api.repository.postgres import PostgresTaskRepository


DSN = os.getenv("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DSN, reason="set DATABASE_URL for live Postgres")


@pytest.fixture(scope="module")
def event_loop_policy():
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


async def test_rich_repository_writes_the_canonical_graph() -> None:
    pool = AsyncConnectionPool(DSN, open=False, kwargs={"autocommit": True})
    await pool.open()
    repo = PostgresTaskRepository(pool)
    thread_id = str(uuid4())
    task = AgentTask(
        id=str(uuid4()), goal="persist graph", thread_id=thread_id,
        track=AgentTrack.LANGGRAPH,
    )
    try:
        await repo.save_task(task)
        run_id = await repo.create_run(
            task.id, ApiSettings().build_agent_config(AgentTrack.LANGGRAPH)
        )
        await repo.mark_run_running(run_id)
        phase_id = await repo.save_phase(run_id, {
            "sequence": 0, "phase": "investigate", "entry_reason": "test",
        })
        step_id = str(uuid4())
        plan_id = await repo.save_plan(run_id, {
            "phase_id": phase_id, "revision": 0, "summary": "inspect",
            "steps": [{"id": step_id, "step_number": 0, "description": "read"}],
        })
        assert plan_id
        await repo.save_finding(run_id, {
            "phase_id": phase_id, "plan_step_id": step_id,
            "description": "port", "detail": "8080",
        })
        await repo.save_verification(run_id, {
            "plan_step_id": step_id, "result": "verified", "deterministic": True,
        })
        await repo.save_trace_ref(run_id, {"trace_id": f"trace-{uuid4()}"})
        approval_id = await repo.save_approval(run_id, {
            "plan_step_id": step_id, "reason": "write requires review",
        })
        await repo.resolve_approval({
            "approval_id": approval_id, "approved": True, "note": "test",
        })
        await repo.close_phase(run_id, {"phase_id": phase_id})

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    (SELECT count(*) FROM run_phases WHERE run_id=%s),
                    (SELECT count(*) FROM plans WHERE run_id=%s),
                    (SELECT count(*) FROM plan_steps WHERE run_id=%s),
                    (SELECT count(*) FROM run_findings WHERE run_id=%s),
                    (SELECT count(*) FROM verification_results WHERE run_id=%s),
                    (SELECT count(*) FROM trace_refs WHERE run_id=%s),
                    (SELECT count(*) FROM approval_requests WHERE run_id=%s AND status='approved')
                """,
                (run_id,) * 7,
            )
            assert await cur.fetchone() == (1, 1, 1, 1, 1, 1, 1)
    finally:
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("DELETE FROM agent_threads WHERE id=%s", (thread_id,))
        await pool.close()

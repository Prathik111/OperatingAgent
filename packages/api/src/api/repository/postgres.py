"""Postgres-backed ``TaskRepository`` over the 5-table run spine.

Writes ``actors -> agent_threads -> agent_tasks -> config_snapshots ->
agent_runs -> agent_events`` using a psycopg 3 async connection pool. The pool
runs in autocommit mode; the multi-row writes (``save_task``, ``create_run``)
are wrapped in an explicit ``conn.transaction()`` so they land atomically.

Only the spine is persisted this pass — plans, tool/llm calls, verifications
and findings are not written. This backend is covered by an opt-in live test
tier (gated on ``DATABASE_URL``), not by the hermetic unit suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg.types.json import Jsonb

from common.agent import AgentRunResult, AgentTask
from common.config import AgentConfig
from common.enums import AgentTrack, RunStatus, TaskStatus
from common.events import LLMCallRecord, ToolCallRecord

from ..errors import TaskNotFound
from ..serialization import config_content_hash, config_to_snapshot
from . import _sql

if TYPE_CHECKING:  # avoid importing psycopg_pool at module import time
    from psycopg_pool import AsyncConnectionPool

#: Stable identity for the actor that owns API-created threads.
_API_ACTOR_EXTERNAL_ID = "system:api"
_API_ACTOR_DISPLAY_NAME = "API service"


class PostgresTaskRepository:
    def __init__(self, pool: "AsyncConnectionPool") -> None:
        self._pool = pool

    async def save_task(self, task: AgentTask) -> None:
        title = task.metadata.get("title")
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        _sql.UPSERT_ACTOR,
                        (_API_ACTOR_EXTERNAL_ID, _API_ACTOR_DISPLAY_NAME),
                    )
                    actor_id = (await cur.fetchone())[0]
                    await cur.execute(
                        _sql.UPSERT_THREAD, (task.thread_id, actor_id, title)
                    )
                    await cur.execute(
                        _sql.INSERT_TASK,
                        (
                            task.id,
                            task.thread_id,
                            task.goal,
                            task.track.value,
                            TaskStatus.PLANNING.value,
                            Jsonb(task.metadata),
                        ),
                    )

    async def get_task(self, task_id: str) -> AgentTask:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_sql.SELECT_TASK, (task_id,))
                row = await cur.fetchone()
        if row is None:
            raise TaskNotFound(task_id)
        id_, thread_id, goal, track, metadata, created_at = row
        return AgentTask(
            id=str(id_),
            goal=goal,
            thread_id=thread_id,
            track=AgentTrack(track),
            metadata=metadata or {},
            created_at=created_at,
        )

    async def create_run(self, task_id: str, config: AgentConfig) -> str:
        snapshot = config_to_snapshot(config)
        content_hash = config_content_hash(snapshot)
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        _sql.UPSERT_CONFIG_SNAPSHOT,
                        (
                            content_hash,
                            Jsonb(snapshot["llm_config"]),
                            Jsonb(snapshot["execution_config"]),
                            Jsonb(snapshot["sandbox_config"]),
                            Jsonb(snapshot["permissions_config"]),
                            Jsonb(snapshot["checkpoint_config"]),
                            Jsonb(snapshot["tracing_config"]),
                            Jsonb(snapshot["behaviour_config"]),
                            Jsonb(snapshot["prompts_config"]),
                        ),
                    )
                    snapshot_id = (await cur.fetchone())[0]
                    await cur.execute(
                        _sql.INSERT_RUN,
                        (task_id, task_id, snapshot_id, RunStatus.CREATED.value),
                    )
                    run_id = (await cur.fetchone())[0]
        return str(run_id)

    async def mark_run_running(self, run_id: str) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_sql.MARK_RUN_RUNNING, (run_id,))

    async def append_event(
        self, run_id: str, event, sequence_number: int
    ) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _sql.INSERT_EVENT,
                    (run_id, sequence_number, event.type, Jsonb(event.payload)),
                )

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _sql.INSERT_LLM_CALL,
                    (
                        run_id, record.node_name, record.provider, record.model,
                        record.prompt_tokens, record.completion_tokens, record.cost,
                        record.error, record.started_at, record.finished_at,
                    ),
                )

    async def upsert_tool(
        self, server_name: str, base_url: str | None, tool_spec: dict
    ) -> str:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    tool_id = await self._upsert_tool(
                        cur, server_name, base_url, tool_spec
                    )
                    return str(tool_id)

    async def _upsert_tool(
        self, cur, server_name: str, base_url: str | None, tool_spec: dict
    ):
        await cur.execute(_sql.UPSERT_MCP_SERVER, (server_name, base_url))
        server_id = (await cur.fetchone())[0]
        await cur.execute(
            _sql.UPSERT_TOOL,
            (
                server_id,
                tool_spec["name"],
                tool_spec.get("description"),
                Jsonb(tool_spec.get("input_schema")),
            ),
        )
        return (await cur.fetchone())[0]

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    tool_id = await self._upsert_tool(
                        cur,
                        record.server_name,
                        record.base_url,
                        {
                            "name": record.tool_name,
                            "description": record.description,
                            "input_schema": record.input_schema,
                        },
                    )
                    await cur.execute(
                        _sql.INSERT_TOOL_CALL,
                        (
                            run_id, tool_id, Jsonb(record.arguments), record.success,
                            Jsonb(record.output), record.error, record.risk_level,
                            record.risk_reason, record.attempt, record.started_at,
                            record.finished_at,
                        ),
                    )

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _sql.FINALIZE_RUN,
                    (
                        result.status.value,
                        result.output,
                        result.metadata.get("error"),
                        run_id,
                    ),
                )

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_sql.UPDATE_TASK_STATUS, (status.value, task_id))

    async def get_latest_run_status(self, task_id: str) -> RunStatus | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_sql.SELECT_LATEST_RUN_STATUS, (task_id,))
                row = await cur.fetchone()
        if row is None:
            return None
        return RunStatus(row[0])

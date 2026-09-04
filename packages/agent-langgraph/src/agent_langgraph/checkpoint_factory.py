"""Checkpointer construction for the LangGraph track.

The checkpointer is what makes the executor→verifier loop resumable: each arc
through the graph is persisted, so a crashed or interrupted run can resume at
the step it reached instead of replaying from the start.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from common.config import AgentConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

log = logging.getLogger(__name__)

_ALLOWED_MSGPACK_MODULES = (
    ("common.enums", "VerificationResult"),
    ("common.enums", "RunStatus"),
    ("common.enums", "TaskStatus"),
    ("common.enums", "WorkflowPhase"),
    ("agent_langgraph.graph.state", "AgentPlan"),
    ("agent_langgraph.graph.state", "Finding"),
    ("agent_langgraph.graph.state", "PlanStep"),
)


def _checkpoint_serializer() -> JsonPlusSerializer:
    """Allow only the application types intentionally persisted in graph state."""
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


class CheckpointFactory:
    """Builds the checkpoint saver described by ``config.checkpoint``."""

    def __init__(self, config: AgentConfig):
        self.config = config

    @asynccontextmanager
    async def open_checkpointer(
        self,
    ) -> AsyncGenerator[BaseCheckpointSaver | None, None]:
        """Open and initialise a saver for the agent lifecycle.

        ``memory`` is process-local and fine for dev/tests. ``postgres`` is the
        durable production backend; if its driver isn't installed this raises
        rather than silently falling back to memory — quietly losing durability
        in production is worse than failing at startup.
        """
        if not self.config.execution.enable_checkpoints:
            log.info("checkpointing disabled by config; running without a saver")
            yield None
            return

        backend = (self.config.checkpoint.backend or "memory").lower()

        if backend in ("memory", "inmemory", "in_memory"):
            yield MemorySaver(serde=_checkpoint_serializer())
            return

        if backend == "postgres":
            async with self._open_postgres_saver() as saver:
                yield saver
            return

        if backend in {"sqlite", "file", "file-based", "file_based"}:
            async with self._open_sqlite_saver() as saver:
                yield saver
            return

        raise ValueError(f"unknown checkpoint backend: {backend!r}")

    @asynccontextmanager
    async def _open_postgres_saver(
        self,
    ) -> AsyncGenerator[BaseCheckpointSaver, None]:
        connection_string = self.config.checkpoint.connection_string
        if not connection_string:
            raise ValueError(
                "checkpoint.backend is 'postgres' but checkpoint.connection_string is not set"
            )

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "checkpoint.backend is 'postgres' but its driver is not installed; "
                "install the 'langgraph-checkpoint-postgres' package"
            ) from exc

        async with AsyncPostgresSaver.from_conn_string(
            connection_string,
            serde=_checkpoint_serializer(),
        ) as saver:
            await saver.setup()
            log.info(
                "using Postgres checkpointer (namespace=%s)",
                self.config.checkpoint.namespace,
            )
            yield saver

    @asynccontextmanager
    async def _open_sqlite_saver(
        self,
    ) -> AsyncGenerator[BaseCheckpointSaver, None]:
        database_path = self.config.checkpoint.connection_string
        if not database_path:
            raise ValueError(
                "checkpoint.backend is 'sqlite' but checkpoint.connection_string is not set"
            )

        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "checkpoint.backend is 'sqlite' but its driver is not installed; "
                "install the 'langgraph-checkpoint-sqlite' package"
            ) from exc

        async with aiosqlite.connect(database_path) as connection:
            saver = AsyncSqliteSaver(
                connection,
                serde=_checkpoint_serializer(),
            )
            await saver.setup()
            log.info(
                "using SQLite checkpointer (path=%s, namespace=%s)",
                database_path,
                self.config.checkpoint.namespace,
            )
            yield saver

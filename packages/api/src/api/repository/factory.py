"""Repository construction from settings.

Mirrors the memory/postgres split of ``agent_langgraph.checkpoint_factory``: an
in-memory store by default, a Postgres store when explicitly configured — and,
like that factory, a Postgres backend without a DSN raises rather than silently
degrading (losing the system of record in production is worse than failing at
startup).

Returns ``(repository, pool_or_none)`` so the app lifespan owns opening and
closing the connection pool; the factory itself does no I/O, which keeps it
callable from tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import TaskRepository
from .memory import InMemoryTaskRepository

if TYPE_CHECKING:
    from ..config import ApiSettings


def build_repository(settings: ApiSettings) -> tuple[TaskRepository, Any | None]:
    backend = (settings.repository_backend or "memory").lower()

    if backend in ("memory", "inmemory", "in_memory"):
        return InMemoryTaskRepository(), None

    if backend == "postgres":
        if not settings.database_url:
            raise ValueError(
                "repository_backend is 'postgres' but DATABASE_URL is not set"
            )
        from psycopg_pool import AsyncConnectionPool

        from .postgres import PostgresTaskRepository

        # open=False: the lifespan awaits pool.open() so a failed connection
        # surfaces at startup, not lazily on the first request.
        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            settings.database_url, open=False, kwargs={"autocommit": True}
        )
        return PostgresTaskRepository(pool), pool

    raise ValueError(f"unknown repository backend: {backend!r}")

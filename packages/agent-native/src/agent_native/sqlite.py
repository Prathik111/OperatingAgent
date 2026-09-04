"""Durable SQLite implementation of the native agent ``Database`` contract.

The native memory database already owns the canonical Python object model.  This
adapter persists those collections as one atomically replaced SQLite row, which
keeps the desktop implementation small while retaining transactions, WAL
journaling, and crash-safe durability.  PostgreSQL remains the server backend.
"""

from __future__ import annotations

import asyncio
import pickle
import sqlite3
from pathlib import Path
from typing import Any

from .database import MemoryDatabase

_TABLE = "native_database_state"
_STATE_FIELDS = (
    "_sessions",
    "_messages",
    "_events",
    "_sequence",
    "_runs",
    "_grants",
    "_memories",
)


class SQLiteDatabase(MemoryDatabase):
    """Durable native ``Database`` backed by a local SQLite file."""

    def __init__(self, database_path: str | Path) -> None:
        super().__init__()
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), state BLOB NOT NULL)"
        )
        self._connection.commit()
        self._write_lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("SQLite database is closed")
        row = connection.execute(
            f"SELECT state FROM {_TABLE} WHERE id = 1"
        ).fetchone()
        if row is None:
            return
        try:
            state = pickle.loads(row[0])
        except (
            EOFError,
            pickle.PickleError,
            TypeError,
            ValueError,
            ImportError,
            AttributeError,
        ) as exc:
            raise RuntimeError(
                f"SQLite native state at {self.database_path} is invalid"
            ) from exc
        if not isinstance(state, dict):
            raise TypeError(
                f"SQLite native state at {self.database_path} has an invalid shape"
            )
        for field in _STATE_FIELDS:
            value = state.get(field)
            if isinstance(value, (dict, list)):
                setattr(self, field, value)

    def _write_sync(self) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("SQLite database is closed")
        state = {field: getattr(self, field) for field in _STATE_FIELDS}
        payload = sqlite3.Binary(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        connection.execute(
            f"INSERT INTO {_TABLE} (id, state) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET state = excluded.state",
            (payload,),
        )
        connection.commit()

    async def _persist_locked(self) -> None:
        await asyncio.to_thread(self._write_sync)

    async def close(self) -> None:
        """Flush and close the SQLite connection."""
        async with self._write_lock:
            if self._connection is not None:
                await self._persist_locked()
                connection = self._connection
                if connection is None:
                    return
                self._connection = None  # type: ignore[assignment]
                await asyncio.to_thread(connection.close)

    async def create_session(self, session: Any) -> None:
        async with self._write_lock:
            await super().create_session(session)
            await self._persist_locked()

    async def delete_session(self, session_id: str) -> bool:
        async with self._write_lock:
            deleted = await super().delete_session(session_id)
            if deleted:
                await self._persist_locked()
            return deleted

    async def save_message(self, message: Any) -> None:
        async with self._write_lock:
            await super().save_message(message)
            await self._persist_locked()

    async def save_event(self, event: Any) -> None:
        async with self._write_lock:
            await super().save_event(event)
            await self._persist_locked()

    async def next_sequence(self, session_id: str) -> int:
        async with self._write_lock:
            value = await super().next_sequence(session_id)
            await self._persist_locked()
            return value

    async def save_run(self, run: Any) -> None:
        async with self._write_lock:
            await super().save_run(run)
            await self._persist_locked()

    async def save_permission(self, grant: Any) -> None:
        async with self._write_lock:
            await super().save_permission(grant)
            await self._persist_locked()

    async def save_memory(self, memory: Any) -> None:
        async with self._write_lock:
            await super().save_memory(memory)
            await self._persist_locked()

    async def touch_memory(self, memory_id: str, when: Any) -> None:
        async with self._write_lock:
            await super().touch_memory(memory_id, when)
            await self._persist_locked()

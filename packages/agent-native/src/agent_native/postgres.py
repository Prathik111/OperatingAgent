"""The same store, but on Postgres, so a run survives the process that made it.

`MemoryDatabase` is perfect until you close the terminal. This is the version
that remembers: same sixteen methods, same promises, different place to put them.
Nothing above `Database` changes - the loop, the event bus and the permission
store cannot tell which one they were handed, which is the whole point of the
interface being small.

Three things are worth knowing before reading the code.

**Messages are stored as JSON, not as tables.** A message is a list of parts
(text, reasoning, a tool call, a compaction summary), and shredding that into
normalised rows would mean a migration every time a new kind of part is invented.
The parts go in a JSONB column and come back through the same factories the rest
of the code uses. The one column that isn't JSON is `ordinal`, because reading a
conversation back in the right order matters more than anything else here.

**The event counter lives in the database.** `next_sequence` is a single
`UPDATE ... RETURNING`, so two runs asking at the same instant get two different
numbers. Doing it in Python would be correct until the day a second process opens
the same database, and then it would be quietly, unfixably wrong.

**asyncpg is imported lazily.** The package still imports on a machine that never
installed it; you only need it if you actually ask for Postgres.

**A dropped connection is retried, not fatal.** Every query goes through `_run`,
which acquires a connection from the pool, runs the query, and - if the
connection was lost (the server bounced, a network blip, the pool handed back a
stale socket) - waits and tries again with a fresh one, a few times, before
giving up. This mirrors the loop's own retry-on-transient-failure: a database
hiccup should cost a pause, not the run.

**The schema is versioned.** `apply_schema` records which migrations have run in
a `schema_migrations` table and applies only the ones that haven't, each in a
transaction. `schema.sql` is the baseline (version 1); later changes append a
`(version, sql)` step to `_MIGRATIONS` rather than editing the baseline, so a
database created months ago moves forward without being rebuilt.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .conversation import (
    Compaction,
    Conversation,
    Message,
    Reasoning,
    Role,
    Session,
    Text,
    ToolCall,
    ToolCallStatus,
    Usage,
)
from .database import Database
from .events import Event
from .memory import Memory
from .permissions import PermissionDuration, PermissionGrant

LOGGER = logging.getLogger(__name__)

#: The schema, shipped next to this file so `apply_schema()` needs no arguments.
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Version stamped for the baseline `schema.sql`. Bump-by-appending: never edit
#: the baseline or a shipped migration, add the next `(version, sql)` step below.
BASELINE_VERSION = 1

#: Ordered schema changes applied *after* the baseline, each recorded once it
#: runs. This is where a future column or table goes, so an existing database can
#: move forward without being dropped and recreated. Keep versions ascending and
#: above the baseline.
#:
#: v2 adds `runs.reasoning_tokens` for Step 21's thinking-budget receipt: how many
#: of a run's output tokens went to reasoning. `IF NOT EXISTS` + a `DEFAULT 0` make
#: it safe on a database that already has the column and on old rows that predate
#: it, so applying the migration twice, or to a fresh baseline, is a no-op.
_MIGRATIONS: "list[tuple[int, str]]" = [
    (2, "ALTER TABLE runs ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER NOT NULL DEFAULT 0"),
]

#: The bookkeeping table the migration runner reads and writes. Created before
#: anything else, and itself idempotent, so a brand-new database and a long-lived
#: one take the same path.
_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class PostgresDatabase(Database):
    """The Database interface, backed by Postgres through asyncpg.

    Build it with a connection string and `await connect()` before use, or use
    `await PostgresDatabase.open(dsn)` which does both and applies the schema.
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 1,
        max_size: int = 10,
        *,
        max_retries: int = 3,
        retry_first_delay: float = 0.5,
    ) -> None:
        self.dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None
        #: How many times a lost connection is retried before the error is raised,
        #: and the first backoff in seconds (it doubles each attempt). The defaults
        #: echo the loop's retry-on-transient philosophy; a caller that would
        #: rather a database hiccup fail fast can pass ``max_retries=0``.
        self._max_retries = max(0, max_retries)
        self._retry_first_delay = max(0.0, retry_first_delay)
        #: Built once by ``_transient_errors`` then cached - see there for why it
        #: can't just be a module constant.
        self._transient: "tuple[type[BaseException], ...] | None" = None

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    async def open(cls, dsn: str, apply_schema: bool = True) -> "PostgresDatabase":
        """Connect, bring the schema up to the current version, and hand back the store."""
        db = cls(dsn)
        await db.connect()
        if apply_schema:
            await db.apply_schema()
        return db

    async def connect(self) -> None:
        if self._pool is not None:
            return
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "Postgres support needs asyncpg. Install it with "
                "`uv sync --all-packages --extra postgres`."
            ) from exc
        self._pool = await asyncpg.create_pool(
            self.dsn, min_size=self._min_size, max_size=self._max_size
        )

    async def apply_schema(self) -> None:
        """Bring the database up to the current schema version, once and for all.

        Idempotent by construction: a ``schema_migrations`` table records which
        versions have run, and only the missing ones are applied - each inside a
        transaction, so a step that fails halfway rolls back instead of leaving a
        half-built schema. The baseline ``schema.sql`` is version 1
        (``BASELINE_VERSION``); every later change is a ``(version, sql)`` step in
        ``_MIGRATIONS``, applied in ascending order. Runs through ``_run``, so a
        dropped connection mid-migration is retried like any other query - and
        because each step is transactional, a retry re-applies only what didn't
        commit.
        """

        async def operation(conn: Any) -> None:
            async with conn.transaction():
                await conn.execute(_MIGRATIONS_TABLE_SQL)
                applied = {
                    row["version"]
                    for row in await conn.fetch("SELECT version FROM schema_migrations")
                }
                if BASELINE_VERSION not in applied:
                    await conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
                    await _record_migration(conn, BASELINE_VERSION)
                    applied.add(BASELINE_VERSION)
                for version, sql in sorted(_MIGRATIONS):
                    if version in applied:
                        continue
                    await conn.execute(sql)
                    await _record_migration(conn, version)

        await self._run(operation)

    async def schema_version(self) -> int:
        """The highest schema version applied (0 on a database that has none yet)."""

        async def operation(conn: Any) -> int:
            await conn.execute(_MIGRATIONS_TABLE_SQL)
            value = await conn.fetchval(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            )
            return int(value or 0)

        return await self._run(operation)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # -- retry-aware execution --------------------------------------------
    def _acquire(self):
        if self._pool is None:
            raise RuntimeError("Not connected. Call `await database.connect()` first.")
        return self._pool.acquire()

    def _transient_errors(self) -> "tuple[type[BaseException], ...]":
        """Exception types worth retrying: a lost or unreachable connection.

        Built once and cached. The builtin socket errors always apply; asyncpg's
        own connection-loss classes are added when it's installed - it's an
        optional dependency whose exception tree has shifted between versions, so
        importing defensively and reading the classes by name beats hard-coding
        one that might not exist. A programming error - bad SQL, wrong argument
        count, a constraint violation - is deliberately *not* here: retrying it
        only spends the backoff on its way to the same failure.
        """
        if self._transient is not None:
            return self._transient
        errors: "list[type[BaseException]]" = [
            ConnectionError,
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
        ]
        try:
            import asyncpg
        except ImportError:  # pragma: no cover - depends on install
            pass
        else:
            for attr in (
                "PostgresConnectionError",
                "InterfaceError",
                "ConnectionDoesNotExistError",
            ):
                exc_type = getattr(asyncpg, attr, None)
                if isinstance(exc_type, type):
                    errors.append(exc_type)
        self._transient = tuple(dict.fromkeys(errors))
        return self._transient

    async def _run(self, operation: Any) -> Any:
        """Run ``operation(conn)`` on a pooled connection, retrying a lost one.

        ``operation`` is a callable taking a connection and returning an awaitable,
        so it can be re-run against a *fresh* connection - the only thing that
        helps, since a dropped connection can't be resumed. A transient failure
        waits and tries again with exponential backoff (the store's echo of the
        loop's retry-on-transient rule); anything else, and the final attempt, is
        raised unchanged. When the retry works it's invisible: callers see
        ordinary asyncpg results and ordinary asyncpg errors.
        """
        delay = self._retry_first_delay
        last: "BaseException | None" = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._acquire() as conn:
                    return await operation(conn)
            except self._transient_errors() as exc:
                last = exc
                if attempt >= self._max_retries:
                    break
                LOGGER.warning(
                    "database connection failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                delay *= 2
        assert last is not None  # only reached after at least one caught failure
        raise last

    async def _execute(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.execute(sql, *args))

    async def _fetch(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.fetch(sql, *args))

    async def _fetchrow(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.fetchrow(sql, *args))

    async def _fetchval(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.fetchval(sql, *args))

    # -- sessions ----------------------------------------------------------
    async def create_session(self, session: Session) -> None:
        await self._execute(
            """
            INSERT INTO sessions (id, agent, title, working_directory, revision)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id) DO NOTHING
            """,
            session.id,
            session.agent,
            session.title,
            session.working_directory,
            session.revision,
        )

    async def get_session(self, session_id: str) -> "Session | None":
        row = await self._fetchrow(
            "SELECT id, agent, title, working_directory, revision "
            "FROM sessions WHERE id = $1",
            session_id,
        )
        if row is None:
            return None
        return Session(
            id=row["id"],
            agent=row["agent"],
            title=row["title"],
            working_directory=row["working_directory"],
            revision=row["revision"],
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete the session and its rows in one transaction; report if it existed.

        No foreign keys carry an ON DELETE CASCADE, so each dependent table is cleared
        explicitly. Session-scoped grants and memories go; the global ones (empty
        session_id) are matched by `= $1` and so are left untouched. The `DELETE FROM
        sessions` status tag ("DELETE <n>") tells us whether a row was actually there.
        """

        async def operation(conn: Any) -> bool:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages WHERE session_id = $1", session_id)
                await conn.execute("DELETE FROM events WHERE session_id = $1", session_id)
                await conn.execute("DELETE FROM runs WHERE session_id = $1", session_id)
                await conn.execute(
                    "DELETE FROM permission_grants WHERE session_id = $1", session_id
                )
                await conn.execute("DELETE FROM memories WHERE session_id = $1", session_id)
                status = await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)
            # asyncpg returns a command tag like "DELETE 1"; the count is the last field.
            try:
                return int(str(status).split()[-1]) > 0
            except (ValueError, IndexError):  # pragma: no cover - defensive
                return False

        return await self._run(operation)

    async def list_sessions(self, working_directory: str = "", limit: int = 0) -> list:
        """Sessions newest first. `created_at DESC` orders; `id` breaks a tie.

        The tie-breaker echoes `list_runs`: two sessions can share a `created_at`
        to the microsecond, and a history view that reorders them between calls is
        one you can't diff.
        """
        sql = (
            "SELECT id, agent, title, working_directory, revision FROM sessions "
        )
        args: list = []
        if working_directory:
            sql += "WHERE working_directory = $1 "
            args.append(working_directory)
        sql += "ORDER BY created_at DESC, id DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"  # int-cast, so no value reaches SQL unchecked
        rows = await self._fetch(sql, *args)
        return [
            Session(
                id=row["id"],
                agent=row["agent"],
                title=row["title"],
                working_directory=row["working_directory"],
                revision=row["revision"],
            )
            for row in rows
        ]
    async def save_message(self, message: Message) -> None:
        await self._execute(
            """
            INSERT INTO messages (id, session_id, role, parts, model, usage, created_at)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7)
            ON CONFLICT (id) DO NOTHING
            """,
            message.id,
            message.session_id,
            message.role.value,
            json.dumps([_part_to_json(p) for p in message.parts]),
            message.model,
            json.dumps(_usage_to_json(message.usage)) if message.usage else None,
            message.created_at,
        )

    async def load_conversation(self, session_id: str) -> Conversation:
        rows = await self._fetch(
            "SELECT id, session_id, role, parts, model, usage, created_at "
            "FROM messages WHERE session_id = $1 ORDER BY ordinal",
            session_id,
        )
        return Conversation([_row_to_message(row) for row in rows])

    # -- events ------------------------------------------------------------
    async def next_sequence(self, session_id: str) -> int:
        """Hand out the next event number. Atomic, so two runs can't collide.

        The counter is bumped and read in one statement. If the session row isn't
        there (an event for a session nobody created), it's inserted first rather
        than dropping the event - losing an event is worse than an orphan row. The
        whole thing is one ``_run`` operation, so a dropped connection retries it
        coherently from the top rather than half-way through.
        """

        async def operation(conn: Any) -> int:
            sequence = await conn.fetchval(
                "UPDATE sessions SET event_sequence = event_sequence + 1 "
                "WHERE id = $1 RETURNING event_sequence",
                session_id,
            )
            if sequence is None:
                await conn.execute(
                    "INSERT INTO sessions (id, event_sequence) VALUES ($1, 0) "
                    "ON CONFLICT (id) DO NOTHING",
                    session_id,
                )
                sequence = await conn.fetchval(
                    "UPDATE sessions SET event_sequence = event_sequence + 1 "
                    "WHERE id = $1 RETURNING event_sequence",
                    session_id,
                )
            return int(sequence)

        return await self._run(operation)

    async def save_event(self, event: Event) -> None:
        await self._execute(
            """
            INSERT INTO events (session_id, sequence, type, run_id, data, time)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            ON CONFLICT (session_id, sequence) DO NOTHING
            """,
            event.session_id,
            event.sequence,
            event.type,
            event.run_id,
            json.dumps(event.data, default=str),
            event.time,
        )

    async def load_events(self, session_id: str, after_sequence: int = 0) -> list:
        rows = await self._fetch(
            "SELECT session_id, sequence, type, run_id, data, time FROM events "
            "WHERE session_id = $1 AND sequence > $2 ORDER BY sequence",
            session_id,
            after_sequence,
        )
        return [
            Event(
                sequence=row["sequence"],
                type=row["type"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                data=_load_json(row["data"], {}),
                time=row["time"],
            )
            for row in rows
        ]

    # -- runs --------------------------------------------------------------
    async def save_run(self, run: Any) -> None:
        """Store (or update) a run's receipt. Upserts, so a re-saved run overwrites."""
        await self._execute(
            """
            INSERT INTO runs (
                run_id, session_id, status, turns, input_tokens, output_tokens,
                cached_tokens, duration_seconds, cost_usd, model, retries, error,
                reasoning_tokens
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (run_id) DO UPDATE SET
                status           = EXCLUDED.status,
                turns            = EXCLUDED.turns,
                input_tokens     = EXCLUDED.input_tokens,
                output_tokens    = EXCLUDED.output_tokens,
                cached_tokens    = EXCLUDED.cached_tokens,
                duration_seconds = EXCLUDED.duration_seconds,
                cost_usd         = EXCLUDED.cost_usd,
                model            = EXCLUDED.model,
                retries          = EXCLUDED.retries,
                error            = EXCLUDED.error,
                reasoning_tokens = EXCLUDED.reasoning_tokens
            """,
            getattr(run, "run_id", ""),
            getattr(run, "session_id", ""),
            str(getattr(run, "status", "")),
            int(getattr(run, "turns", 0) or 0),
            int(getattr(run, "input_tokens", 0) or 0),
            int(getattr(run, "output_tokens", 0) or 0),
            int(getattr(run, "cached_tokens", 0) or 0),
            float(getattr(run, "duration_seconds", 0.0) or 0.0),
            float(getattr(run, "cost_usd", 0.0) or 0.0),
            str(getattr(run, "model", "") or ""),
            int(getattr(run, "retries", 0) or 0),
            str(getattr(run, "error", "") or ""),
            int(getattr(run, "reasoning_tokens", 0) or 0),
        )

    async def get_run(self, run_id: str) -> Any:
        """One run's receipt, rebuilt into a `RunRecord`, or None if there isn't one."""
        row = await self._fetchrow(
            "SELECT run_id, session_id, status, turns, input_tokens, output_tokens, "
            "cached_tokens, duration_seconds, cost_usd, model, retries, error, "
            "reasoning_tokens "
            "FROM runs WHERE run_id = $1",
            run_id,
        )
        return _row_to_run(row) if row is not None else None

    async def list_runs(self, session_id: str = "", limit: int = 0) -> list:
        """Receipts newest first. `created_at DESC` is the order; `run_id` breaks a tie.

        The tie-breaker matters for the same reason `messages.ordinal` does: two
        runs can share a `created_at` to the microsecond, and a report that lists
        them in a shuffling order every time is not a report you can diff.
        """
        sql = (
            "SELECT run_id, session_id, status, turns, input_tokens, output_tokens, "
            "cached_tokens, duration_seconds, cost_usd, model, retries, error, "
            "reasoning_tokens FROM runs "
        )
        args: list = []
        if session_id:
            sql += "WHERE session_id = $1 "
            args.append(session_id)
        sql += "ORDER BY created_at DESC, run_id DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"  # int-cast, so no value reaches SQL unchecked
        rows = await self._fetch(sql, *args)
        return [_row_to_run(row) for row in rows]

    # -- permissions -------------------------------------------------------
    async def save_permission(self, grant: Any) -> None:
        await self._execute(
            "INSERT INTO permission_grants "
            "(tool_pattern, duration, session_id, argument_pattern) "
            "VALUES ($1, $2, $3, $4)",
            getattr(grant, "tool_pattern", ""),
            _duration_value(getattr(grant, "duration", PermissionDuration.ONCE)),
            getattr(grant, "session_id", "") or "",
            getattr(grant, "argument_pattern", "") or "",
        )

    async def load_permissions(self, session_id: str) -> list:
        """Grants for this session, plus the global ones (empty session_id)."""
        rows = await self._fetch(
            "SELECT tool_pattern, duration, session_id, argument_pattern "
            "FROM permission_grants WHERE session_id = $1 OR session_id = ''",
            session_id,
        )
        return [
            PermissionGrant(
                tool_pattern=row["tool_pattern"],
                duration=PermissionDuration(row["duration"]),
                session_id=row["session_id"],
                argument_pattern=row["argument_pattern"] or "",
            )
            for row in rows
        ]

    # -- memories ----------------------------------------------------------
    async def save_memory(self, memory: Memory) -> None:
        """Store a note. Re-saving the same id updates the text rather than duplicating."""
        await self._execute(
            """
            INSERT INTO memories (id, session_id, kind, text, created_at, last_used_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO UPDATE SET
                kind         = EXCLUDED.kind,
                text         = EXCLUDED.text,
                last_used_at = EXCLUDED.last_used_at
            """,
            memory.id,
            memory.session_id or "",
            memory.kind,
            memory.text,
            memory.created_at,
            memory.last_used_at,
        )

    async def load_memories(self, session_id: str = "") -> list:
        """This session's notes plus the unscoped ones. Scoring happens in Python."""
        rows = await self._fetch(
            "SELECT id, session_id, kind, text, created_at, last_used_at "
            "FROM memories WHERE session_id = '' OR $1 = '' OR session_id = $1 "
            "ORDER BY last_used_at DESC",
            session_id or "",
        )
        return [
            Memory(
                id=row["id"],
                session_id=row["session_id"],
                kind=row["kind"],
                text=row["text"],
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
            )
            for row in rows
        ]

    async def touch_memory(self, memory_id: str, when: datetime) -> None:
        await self._execute(
            "UPDATE memories SET last_used_at = $2 WHERE id = $1", memory_id, when
        )


# ---------------------------------------------------------------------------
# Turning messages into JSON and back
# ---------------------------------------------------------------------------
def _part_to_json(part: Any) -> dict:
    """One message part as a plain dict. `kind` is what tells them apart on the way back."""
    if isinstance(part, Text):
        return {"kind": "text", "text": part.text}
    if isinstance(part, Reasoning):
        return {"kind": "reasoning", "text": part.text, "hidden": part.hidden}
    if isinstance(part, ToolCall):
        return {
            "kind": "tool_call",
            "id": part.id,
            "name": part.name,
            "arguments": part.arguments,
            "status": part.status.value,
            "output": part.output,
            "error": part.error,
        }
    if isinstance(part, Compaction):
        return {
            "kind": "compaction",
            "summary": part.summary,
            "old_messages": part.old_messages,
            "tokens_before": part.tokens_before,
            "tokens_after": part.tokens_after,
        }
    # An unknown part is stored as its text rather than dropped: a conversation
    # that loses a message is worse than one that loses some formatting.
    return {"kind": "text", "text": str(getattr(part, "text", ""))}


def _part_from_json(data: dict) -> Any:
    kind = data.get("kind", "text")
    if kind == "reasoning":
        return Reasoning(data.get("text", ""), hidden=data.get("hidden", True))
    if kind == "tool_call":
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments") or {},
            status=ToolCallStatus(data.get("status", "pending")),
            output=data.get("output", "") or "",
            error=data.get("error", "") or "",
        )
    if kind == "compaction":
        return Compaction(
            summary=data.get("summary", ""),
            old_messages=data.get("old_messages") or [],
            tokens_before=data.get("tokens_before", 0),
            tokens_after=data.get("tokens_after", 0),
        )
    return Text(data.get("text", ""))


def _usage_to_json(usage: "Usage | None") -> "dict | None":
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": usage.cached_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _row_to_message(row: Any) -> Message:
    usage_data = _load_json(row["usage"], None)
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=Role(row["role"]),
        parts=[_part_from_json(p) for p in _load_json(row["parts"], [])],
        model=row["model"],
        usage=Usage(**usage_data) if usage_data else None,
        created_at=row["created_at"] or datetime.now(timezone.utc),
    )


def _row_to_run(row: Any) -> Any:
    """A `runs` row back into a `RunRecord`.

    `RunRecord` is imported here rather than at module top so `postgres.py` keeps
    importing on a machine that never pulls in the loop, matching the lazy-asyncpg
    rule above. `created_at` is read only for ordering, so it has no field here.
    """
    from .loop import RunRecord

    return RunRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        status=row["status"],
        turns=row["turns"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        error=row["error"] or "",
        cached_tokens=row["cached_tokens"],
        duration_seconds=row["duration_seconds"],
        cost_usd=row["cost_usd"],
        model=row["model"] or "",
        retries=row["retries"],
        reasoning_tokens=row["reasoning_tokens"],
    )


def _load_json(value: Any, default: Any) -> Any:
    """asyncpg hands back JSONB as a string unless a codec is set. Accept both."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _duration_value(duration: Any) -> str:
    return duration.value if isinstance(duration, PermissionDuration) else str(duration)


async def _record_migration(conn: Any, version: int) -> None:
    """Stamp a schema version as applied. Idempotent within the transaction, so a
    retry that re-enters the same step is harmless."""
    await conn.execute(
        "INSERT INTO schema_migrations (version) VALUES ($1) "
        "ON CONFLICT (version) DO NOTHING",
        version,
    )

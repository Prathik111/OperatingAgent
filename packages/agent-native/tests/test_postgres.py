"""The Postgres store, exercised without a Postgres.

asyncpg isn't installed here and there's no server to talk to, so a tiny fake pool
stands in for both. That's enough to pin the parts that are logic rather than SQL:
the migration runner applies the baseline once and stays idempotent, the retry
wrapper tells a dropped connection apart from a bad query, and a message survives
the round trip through JSON. The SQL itself is checked against a real database on
the user's machine, not here.
"""

from __future__ import annotations

from agent_native.conversation import (
    Compaction,
    Reasoning,
    Role,
    Text,
    ToolCall,
    ToolCallStatus,
    Usage,
)
from agent_native.postgres import (
    BASELINE_VERSION,
    PostgresDatabase,
    _load_json,
    _part_from_json,
    _part_to_json,
    _row_to_message,
    _usage_to_json,
)


# ---------------------------------------------------------------------------
# A fake pool + connection: records SQL, can be told to drop connections
# ---------------------------------------------------------------------------
class _FakeTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, pool: "_FakePool") -> None:
        self._pool = pool

    async def execute(self, sql: str, *args):
        self._pool.calls.append(("execute", sql, args))
        if "INSERT INTO schema_migrations" in sql and args:
            self._pool.applied.add(args[0])
        return "OK"

    async def fetch(self, sql: str, *args):
        self._pool.calls.append(("fetch", sql, args))
        if "FROM schema_migrations" in sql:
            return [{"version": v} for v in sorted(self._pool.applied)]
        return []

    async def fetchrow(self, sql: str, *args):
        self._pool.calls.append(("fetchrow", sql, args))
        return None

    async def fetchval(self, sql: str, *args):
        self._pool.calls.append(("fetchval", sql, args))
        if "MAX(version)" in sql:
            return max(self._pool.applied) if self._pool.applied else 0
        if self._pool.fetchvals:
            return self._pool.fetchvals.pop(0)
        return None

    def transaction(self):
        return _FakeTx()


class _Acquire:
    def __init__(self, pool: "_FakePool") -> None:
        self._pool = pool

    async def __aenter__(self):
        self._pool.acquire_attempts += 1
        if self._pool.acquire_attempts <= self._pool.fail_first:
            raise self._pool.fail_exc("connection reset by peer")
        return _FakeConn(self._pool)

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    """Enough of an asyncpg pool for `PostgresDatabase._run` to drive it.

    `fail_first` acquires raise `fail_exc` before one succeeds - that's how a
    dropped connection is simulated. `fetchvals` is a queue of values handed back
    by `fetchval` for the non-migration queries (the event counter, mostly).
    """

    def __init__(self, fail_first: int = 0, fail_exc=ConnectionError, fetchvals=None) -> None:
        self.calls: list = []
        self.applied: set = set()
        self.acquire_attempts = 0
        self.fail_first = fail_first
        self.fail_exc = fail_exc
        self.fetchvals = list(fetchvals or [])

    def acquire(self):
        return _Acquire(self)

    async def close(self):
        pass


def _db(**pool_kwargs) -> PostgresDatabase:
    # retry_first_delay=0 so a retry doesn't actually sleep during the test.
    db = PostgresDatabase("postgres://test/db", max_retries=3, retry_first_delay=0.0)
    db._pool = _FakePool(**pool_kwargs)
    return db


def _executed(db: PostgresDatabase) -> str:
    return "\n".join(sql for op, sql, _ in db._pool.calls if op == "execute")


def _baseline_runs(db: PostgresDatabase) -> int:
    # 'IF NOT EXISTS sessions' is unique to schema.sql (the ledger DDL names
    # schema_migrations), so counting it counts how often the baseline was applied.
    return sum(sql.count("IF NOT EXISTS sessions") for op, sql, _ in db._pool.calls if op == "execute")


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------
async def test_apply_schema_creates_the_ledger_and_the_baseline():
    db = _db()
    await db.apply_schema()
    assert "schema_migrations" in _executed(db)      # the bookkeeping table first
    assert _baseline_runs(db) == 1                    # schema.sql ran once
    assert BASELINE_VERSION in db._pool.applied       # and was recorded


async def test_apply_schema_is_idempotent():
    db = _db()
    await db.apply_schema()
    await db.apply_schema()                           # a second run on the same db
    assert _baseline_runs(db) == 1                     # baseline is not applied twice


async def test_schema_version_reads_the_highest_recorded():
    db = _db()
    assert await db.schema_version() == 0             # nothing applied yet
    await db.apply_schema()
    assert await db.schema_version() == BASELINE_VERSION


# ---------------------------------------------------------------------------
# The retry wrapper: a blip is retried, a bug is not
# ---------------------------------------------------------------------------
async def test_a_dropped_connection_is_retried_then_succeeds():
    db = _db(fail_first=2, fetchvals=[42])            # first two acquires fail
    value = await db._fetchval("SELECT 1")
    assert value == 42
    assert db._pool.acquire_attempts == 3             # two failures, one success


async def test_a_programming_error_is_not_retried():
    db = _db()

    async def bad(conn):
        raise ValueError("column does not exist")     # a bug, not a blip

    raised = False
    try:
        await db._run(bad)
    except ValueError:
        raised = True
    assert raised
    assert db._pool.acquire_attempts == 1             # tried exactly once


async def test_retries_give_up_after_the_limit_and_raise():
    db = PostgresDatabase("postgres://x", max_retries=2, retry_first_delay=0.0)
    db._pool = _FakePool(fail_first=99)               # never recovers

    raised = False
    try:
        await db._fetchval("SELECT 1")
    except ConnectionError:
        raised = True
    assert raised
    assert db._pool.acquire_attempts == 3             # 1 try + 2 retries


# ---------------------------------------------------------------------------
# The event counter's insert-then-number fallback
# ---------------------------------------------------------------------------
async def test_next_sequence_returns_the_updated_counter():
    db = _db(fetchvals=[5])                            # the UPDATE ... RETURNING
    assert await db.next_sequence("s") == 5


async def test_next_sequence_inserts_a_missing_session_then_numbers_it():
    # The first UPDATE finds no row (None), so a session is inserted and the UPDATE
    # runs again, this time returning 1 - an event is never dropped for want of a
    # session row.
    db = _db(fetchvals=[None, 1])
    assert await db.next_sequence("s") == 1
    assert any(
        "INSERT INTO sessions" in sql for op, sql, _ in db._pool.calls if op == "execute"
    )


# ---------------------------------------------------------------------------
# Messages survive the round trip through JSON
# ---------------------------------------------------------------------------
def test_message_parts_round_trip_through_json():
    parts = [
        Text("hello world"),
        Reasoning("thinking", hidden=True),
        ToolCall(
            id="c1", name="read_file", arguments={"path": "a.txt"},
            status=ToolCallStatus.SUCCESS, output="port=8080", error="",
        ),
        Compaction(summary="did stuff", old_messages=["m1", "m2"], tokens_before=100, tokens_after=10),
    ]
    back = [_part_from_json(_part_to_json(p)) for p in parts]

    assert isinstance(back[0], Text) and back[0].text == "hello world"
    assert isinstance(back[1], Reasoning) and back[1].hidden is True and back[1].text == "thinking"
    call = back[2]
    assert isinstance(call, ToolCall) and call.name == "read_file"
    assert call.arguments == {"path": "a.txt"} and call.output == "port=8080"
    assert call.status == ToolCallStatus.SUCCESS
    comp = back[3]
    assert isinstance(comp, Compaction) and comp.old_messages == ["m1", "m2"]
    assert comp.tokens_before == 100 and comp.tokens_after == 10


def test_an_unknown_part_degrades_to_text_rather_than_being_dropped():
    class _Weird:
        text = "unchanged content"

    data = _part_to_json(_Weird())
    assert data == {"kind": "text", "text": "unchanged content"}
    assert isinstance(_part_from_json(data), Text)


def test_usage_json_round_trips_and_none_stays_none():
    assert _usage_to_json(None) is None
    data = _usage_to_json(Usage(input_tokens=7, output_tokens=3, cached_tokens=1))
    assert data == {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 1}


def test_load_json_accepts_a_string_a_dict_or_garbage():
    assert _load_json('{"a": 1}', {}) == {"a": 1}      # asyncpg hands JSONB back as text
    assert _load_json({"a": 1}, {}) == {"a": 1}         # ...or already decoded
    assert _load_json(None, "default") == "default"     # a null column
    assert _load_json("not json", {}) == {}             # falls back, never raises


def test_row_to_message_rebuilds_role_parts_and_usage():
    call = ToolCall(
        id="c", name="read_file", arguments={"path": "x"},
        status=ToolCallStatus.SUCCESS, output="ok", error="",
    )
    row = {
        "id": "m1",
        "session_id": "s",
        "role": "assistant",
        "parts": [_part_to_json(Text("hello")), _part_to_json(call)],
        "model": "llama-3.3-70b",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 2},
        "created_at": None,
    }
    message = _row_to_message(row)

    assert message.role == Role.ASSISTANT and message.model == "llama-3.3-70b"
    assert message.usage.input_tokens == 10 and message.usage.cached_tokens == 2
    assert [type(p).__name__ for p in message.parts] == ["Text", "ToolCall"]

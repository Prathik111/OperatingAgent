"""Where everything is kept.

`Database` is a small interface so the rest of the agent never knows whether it
is talking to memory or Postgres. `MemoryDatabase` is here; the Postgres one is
in `postgres.py`, behind the same sixteen methods. The important promise is that
events are stored append-only and in order, because that is what lets a client
that dropped off reconnect and ask for "everything after number N".

Run receipts go in through `save_run` and come back out through `get_run` and
`list_runs`. That read-back is what lets a report - or the evaluation harness -
ask "how did this run go?" after the process that made it is gone, rather than
the receipt being write-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .conversation import Conversation, Message, Session
    from .events import Event
    from .memory import Memory


def _utc(value: datetime) -> datetime:
    """Normalize legacy naive UTC values before comparing or storing them."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalize_session(session: Any, messages: list[Any] | None = None) -> Any:
    """Upgrade sessions pickled before timestamp fields were introduced."""
    message_times = [
        _utc(value)
        for message in messages or []
        if isinstance((value := getattr(message, "created_at", None)), datetime)
    ]
    now = datetime.now(UTC)
    raw_created = getattr(session, "created_at", None)
    created_at = (
        _utc(raw_created)
        if isinstance(raw_created, datetime)
        else min(message_times, default=now)
    )
    raw_updated = getattr(session, "updated_at", None)
    updated_at = (
        _utc(raw_updated) if isinstance(raw_updated, datetime) else created_at
    )
    if message_times:
        updated_at = max(updated_at, *message_times)
    session.created_at = created_at
    session.updated_at = max(created_at, updated_at)
    return session


class Database(ABC):
    """The one interface the agent stores things through."""

    @abstractmethod
    async def create_session(self, session: Session) -> None: ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Remove a session and everything filed under it, and say whether it existed.

        Messages, events, runs, and the session's own permission grants and memories
        all go; grants and memories with an empty ``session_id`` are global ("always"
        / cross-session notes) and are deliberately left alone. Returns True if a
        session with that id was present, False if there was nothing to delete - which
        is what lets a caller answer "no such session" rather than pretend it worked.
        """

    @abstractmethod
    async def list_sessions(self, working_directory: str = "", limit: int = 0) -> list:
        """Sessions, newest first.

        An empty ``working_directory`` returns sessions across every folder; a
        specific one narrows to the sessions opened in that folder, matched exactly
        as stored (the caller resolves the path first). ``limit`` of 0 means no cap,
        otherwise at most that many (the most recent). This is the read the run
        history view leans on to turn a ``--dir`` into the set of sessions whose
        runs it then lists.
        """

    @abstractmethod
    async def save_message(self, message: Message) -> None: ...

    @abstractmethod
    async def load_conversation(self, session_id: str) -> Conversation: ...

    @abstractmethod
    async def save_event(self, event: Event) -> None: ...

    @abstractmethod
    async def load_events(self, session_id: str, after_sequence: int = 0) -> list: ...

    @abstractmethod
    async def next_sequence(self, session_id: str) -> int:
        """Hand out the next event number for a session, counting from 1."""

    @abstractmethod
    async def save_run(self, run: Any) -> None: ...

    @abstractmethod
    async def get_run(self, run_id: str) -> Any:
        """The receipt for one run, or None if no run has that id."""

    @abstractmethod
    async def list_runs(self, session_id: str = "", limit: int = 0) -> list:
        """Run receipts, newest first.

        An empty ``session_id`` returns runs across every session; a specific one
        narrows to that session. ``limit`` of 0 means no cap, otherwise at most
        that many (the most recent). This is the read half of ``save_run`` - the
        piece a report or the evaluation harness leans on to see how runs went.
        """

    @abstractmethod
    async def save_permission(self, grant: Any) -> None: ...

    @abstractmethod
    async def load_permissions(self, session_id: str) -> list: ...

    @abstractmethod
    async def save_memory(self, memory: Memory) -> None:
        """Keep a note that should outlive the run that wrote it."""

    @abstractmethod
    async def load_memories(self, session_id: str = "") -> list:
        """Notes for this session plus the ones kept with no session at all.

        Everything comes back and the matching is done in Python, in `memory.py`,
        so both stores rank the same query the same way.
        """

    @abstractmethod
    async def touch_memory(self, memory_id: str, when: datetime) -> None:
        """Record that a note just proved useful, so it stays near the top."""

    async def close(self) -> None:
        """Let go of any resources. No-op for memory; closes the pool for Postgres."""
        return


class MemoryDatabase(Database):
    """Everything in dicts. Fast, disposable, perfect for tests and a single desktop run."""

    def __init__(self) -> None:
        self._sessions: dict = {}
        self._messages: dict = {}      # session_id -> list[Message]
        self._events: dict = {}        # session_id -> list[Event]
        self._sequence: dict = {}      # session_id -> last handed-out number
        self._runs: dict = {}          # run_id -> run record
        self._grants: list = []        # list of permission grants
        self._memories: dict = {}      # memory id -> Memory

    async def create_session(self, session: Session) -> None:
        _normalize_session(session)
        self._sessions[session.id] = session
        self._messages.setdefault(session.id, [])
        self._events.setdefault(session.id, [])
        self._sequence.setdefault(session.id, 0)

    async def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return _normalize_session(session, self._messages.get(session_id, []))

    async def delete_session(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)
        self._events.pop(session_id, None)
        self._sequence.pop(session_id, None)
        self._runs = {
            rid: run for rid, run in self._runs.items()
            if getattr(run, "session_id", "") != session_id
        }
        # Session-scoped grants and notes go; the global ones (empty session_id) stay.
        self._grants = [g for g in self._grants if (getattr(g, "session_id", "") or "") != session_id]
        self._memories = {
            mid: m for mid, m in self._memories.items()
            if (getattr(m, "session_id", "") or "") != session_id
        }
        return existed

    async def list_sessions(self, working_directory: str = "", limit: int = 0) -> list:
        # Match Postgres: recent message activity moves a session to the top.
        sessions = list(self._sessions.values())
        if working_directory:
            sessions = [
                s for s in sessions if getattr(s, "working_directory", "") == working_directory
            ]
        for session in sessions:
            _normalize_session(session, self._messages.get(session.id, []))
        sessions.sort(key=lambda s: (s.updated_at, s.id), reverse=True)
        if limit and limit > 0:
            sessions = sessions[:limit]
        return sessions

    async def save_message(self, message: Message) -> None:
        message.created_at = _utc(message.created_at)
        session = self._sessions.get(message.session_id)
        if session is not None:
            _normalize_session(session, self._messages.get(message.session_id, []))
            session.updated_at = max(session.updated_at, message.created_at)
        self._messages.setdefault(message.session_id, []).append(message)

    async def load_conversation(self, session_id: str) -> Conversation:
        from .conversation import Conversation

        return Conversation(list(self._messages.get(session_id, [])))

    async def save_event(self, event: Event) -> None:
        self._events.setdefault(event.session_id, []).append(event)

    async def load_events(self, session_id: str, after_sequence: int = 0) -> list:
        events = [e for e in self._events.get(session_id, []) if e.sequence > after_sequence]
        events.sort(key=lambda e: e.sequence)  # replay is always in sequence order
        return events

    async def next_sequence(self, session_id: str) -> int:
        nxt = self._sequence.get(session_id, 0) + 1
        self._sequence[session_id] = nxt
        return nxt

    async def save_run(self, run: Any) -> None:
        run_id = getattr(run, "run_id", None) or getattr(run, "id", None) or str(len(self._runs))
        self._runs[run_id] = run

    async def get_run(self, run_id: str) -> Any:
        return self._runs.get(run_id)

    async def list_runs(self, session_id: str = "", limit: int = 0) -> list:
        # Dict insertion order is chronological here, so reversing gives newest
        # first - the same order the Postgres store gets from `created_at DESC`.
        runs = list(self._runs.values())
        if session_id:
            runs = [r for r in runs if getattr(r, "session_id", "") == session_id]
        runs.reverse()
        if limit and limit > 0:
            runs = runs[:limit]
        return runs

    async def save_permission(self, grant: Any) -> None:
        self._grants.append(grant)

    async def load_permissions(self, session_id: str) -> list:
        # Grants scoped to this session, plus grants with no session (global "always").
        out = []
        for g in self._grants:
            gsid = getattr(g, "session_id", "") or ""
            if gsid == "" or gsid == session_id:
                out.append(g)
        return out

    async def save_memory(self, memory: Memory) -> None:
        self._memories[memory.id] = memory

    async def load_memories(self, session_id: str = "") -> list:
        # Same rule as permissions: this session's notes, plus the unscoped ones.
        return [
            m
            for m in self._memories.values()
            if not m.session_id or not session_id or m.session_id == session_id
        ]

    async def touch_memory(self, memory_id: str, when: datetime) -> None:
        memory = self._memories.get(memory_id)
        if memory is not None:
            memory.last_used_at = when

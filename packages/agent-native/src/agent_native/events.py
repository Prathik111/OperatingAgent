"""Numbered events: the running commentary of a session.

Every interesting thing that happens - a message was added, the assistant typed
a few more words, a tool started, the user was asked for permission, the run
finished - becomes an `Event` with a number. The numbers only ever go up, and
every event is stored, so a client that reconnects just says "I last saw 412,
what happened after that?" and gets caught up.

That same number is what a desktop app sends as `Last-Event-ID` to resume an
interrupted stream. Getting ordering and storage right here is what makes resume
and replay possible everywhere else.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database
    from .redaction import Redactor


class EventType:
    """The kinds of events the agent emits. Plain strings so they survive storage."""

    MESSAGE_ADDED = "message_added"
    ASSISTANT_DELTA = "assistant_delta"        # a chunk of streamed text
    REASONING_DELTA = "reasoning_delta"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    PERMISSION_REQUESTED = "permission_requested"
    PERMISSION_RESOLVED = "permission_resolved"
    TURN_STARTED = "turn_started"
    RUN_FINISHED = "run_finished"
    ERROR = "error"
    MODEL_FALLBACK = "model_fallback"  # the run switched to an alternate model


@dataclass
class Event:
    """One numbered thing that happened."""

    sequence: int
    type: str
    session_id: str
    run_id: str = ""
    data: dict = field(default_factory=dict)
    time: datetime = field(default_factory=lambda: datetime.now(UTC))


_STOP = object()  # pushed to a subscriber's queue to end its stream


class EventBus:
    """Hands out numbers, stores every event, and streams them to live listeners."""

    def __init__(self, database: Database, redactor: Redactor | None = None) -> None:
        self._db = database
        self._subscribers: dict = {}  # session_id -> list[asyncio.Queue]
        self._locks: dict[str, asyncio.Lock] = {}
        # Every event's data passes through here before it is stored or streamed,
        # so a secret in a tool's output or an error can't reach an event row, a
        # replayed stream, or a log that prints from it. None means no redaction -
        # the runtime always installs one; a bare EventBus in a test stays literal.
        self._redactor = redactor

    async def _publish_unlocked(self, event: Event) -> None:
        """Save event and fan out without acquiring the session lock. Caller must hold lock."""
        await self._db.save_event(event)
        for queue in self._subscribers.get(event.session_id, []):
            queue.put_nowait(event)

    async def emit(
        self,
        session_id: str,
        type: str,
        data: dict | None = None,
        run_id: str = "",
    ) -> Event:
        """Build the next event for a session, store it, and fan it out. The common path."""
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            sequence = await self._db.next_sequence(session_id)
            clean = data or {}
            if self._redactor is not None:
                clean = self._redactor.redact(clean)
            event = Event(sequence, type, session_id, run_id, clean)
            await self._publish_unlocked(event)
            return event

    async def publish(self, event: Event) -> None:
        """Store an already-numbered event and push it to any live listeners."""
        lock = self._locks.setdefault(event.session_id, asyncio.Lock())
        async with lock:
            await self._publish_unlocked(event)

    async def subscribe(self, session_id: str, from_sequence: int = 0):
        """Yield events for a session: first the stored ones after `from_sequence`,
        then live ones as they happen. Register-before-replay means nothing is
        missed in the gap, and de-duping by number means nothing arrives twice."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(session_id, []).append(queue)
        last = from_sequence
        try:
            for event in await self._db.load_events(session_id, from_sequence):
                if event.sequence <= last:
                    continue
                last = event.sequence
                yield event
            while True:
                event = await queue.get()
                if event is _STOP:
                    return
                if event.sequence <= last:
                    continue
                last = event.sequence
                yield event
        finally:
            listeners = self._subscribers.get(session_id, [])
            if queue in listeners:
                listeners.remove(queue)

    async def close(self) -> None:
        """End every open subscription cleanly."""
        for listeners in self._subscribers.values():
            for queue in listeners:
                queue.put_nowait(_STOP)

"""An in-process pub/sub broker that fans one run's events out to many readers.

Both the SSE endpoint and the WebSocket endpoint subscribe here; the
``TaskService`` publishes here as the orchestrator emits events. It is
single-event-loop and lock-guarded — no threads.

The load-bearing invariant is the **late-subscriber replay**: a client that
connects after a run has already emitted some events must still see them.
:meth:`subscribe` snapshots the replay buffer and registers its queue *under the
same lock*, so no event can slip between the snapshot and the first ``get`` —
it either lands in the snapshot or the queue, never both, never neither.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from common.events import AgentEvent

#: Sentinel pushed to every subscriber queue when a topic closes, so a
#: subscriber's ``await queue.get()`` wakes and the generator returns.
_CLOSE = object()


@dataclass(slots=True)
class _Topic:
    """State for one task_id: the replay buffer, live subscribers, closed flag."""

    buffer: deque[AgentEvent]
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    closed: bool = False


class EventBroker:
    """Fan-out of ``AgentEvent``s keyed by task id, with bounded replay."""

    def __init__(self, *, buffer_size: int = 1000) -> None:
        self._topics: dict[str, _Topic] = {}
        self._lock = asyncio.Lock()
        self._buffer_size = buffer_size

    def _ensure_topic(self, task_id: str) -> _Topic:
        """Get or create a topic. Caller must hold ``self._lock``."""
        topic = self._topics.get(task_id)
        if topic is None:
            topic = _Topic(buffer=deque(maxlen=self._buffer_size))
            self._topics[task_id] = topic
        return topic

    async def publish(self, task_id: str, event: AgentEvent) -> None:
        """Append ``event`` to the replay buffer and hand it to live subscribers."""
        async with self._lock:
            topic = self._ensure_topic(task_id)
            if topic.closed:
                return
            topic.buffer.append(event)
            for queue in topic.subscribers:
                queue.put_nowait(event)

    async def close(self, task_id: str) -> None:
        """Mark a topic terminal and wake every subscriber so its stream ends."""
        async with self._lock:
            topic = self._ensure_topic(task_id)
            topic.closed = True
            for queue in topic.subscribers:
                queue.put_nowait(_CLOSE)

    async def aclose_all(self) -> None:
        """Close every topic — used on application shutdown."""
        async with self._lock:
            for topic in self._topics.values():
                topic.closed = True
                for queue in topic.subscribers:
                    queue.put_nowait(_CLOSE)

    async def subscribe(self, task_id: str) -> AsyncIterator[AgentEvent]:
        """Yield buffered events, then live events, until the topic closes.

        If the topic is already closed when we subscribe, the buffer holds the
        complete history (including the terminal event) — replay it and stop
        without registering a queue, since no further sentinel would ever come.
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            topic = self._ensure_topic(task_id)
            snapshot = list(topic.buffer)
            already_closed = topic.closed
            if not already_closed:
                topic.subscribers.add(queue)

        try:
            for event in snapshot:
                yield event
            if already_closed:
                return
            while True:
                item = await queue.get()
                if item is _CLOSE:
                    return
                yield item
        finally:
            async with self._lock:
                topic = self._topics.get(task_id)
                if topic is not None:
                    topic.subscribers.discard(queue)

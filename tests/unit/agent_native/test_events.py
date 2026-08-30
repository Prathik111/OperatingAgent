"""The event bus: numbered, stored, replayable, and live.

Offline, against the in-memory Database. These pin the two promises the rest of
the system leans on. First, numbers only go up and every event is stored, so a
client that dropped off can say "everything after N?" and be caught up. Second, a
live subscriber sees new events as they happen and never sees one twice - which is
what makes an interrupted desktop stream safe to resume.
"""

from __future__ import annotations

import asyncio

from agent_native.database import MemoryDatabase
from agent_native.events import EventBus, EventType


def _bus() -> EventBus:
    return EventBus(MemoryDatabase())


# -- numbering and storage --------------------------------------------------
async def test_emit_numbers_from_one_and_stores_in_order():
    bus = _bus()
    e1 = await bus.emit("s", EventType.TURN_STARTED)
    e2 = await bus.emit("s", EventType.ASSISTANT_DELTA, {"text": "hi"})
    e3 = await bus.emit("s", EventType.RUN_FINISHED)

    assert [e1.sequence, e2.sequence, e3.sequence] == [1, 2, 3]
    assert e2.data == {"text": "hi"}
    stored = await bus._db.load_events("s")
    assert [e.sequence for e in stored] == [1, 2, 3]


async def test_sequences_are_counted_per_session():
    bus = _bus()
    a = await bus.emit("a", EventType.TURN_STARTED)
    b = await bus.emit("b", EventType.TURN_STARTED)
    assert a.sequence == 1 and b.sequence == 1     # two sessions, two counters


# -- replay -----------------------------------------------------------------
async def test_subscribe_replays_only_events_after_the_given_sequence():
    bus = _bus()
    for _ in range(3):
        await bus.emit("s", EventType.ASSISTANT_DELTA)

    # A client that last saw #1 wants 2 and 3, in order, once each.
    got = []
    agen = bus.subscribe("s", from_sequence=1)
    try:
        for _ in range(2):
            got.append(await asyncio.wait_for(agen.__anext__(), 1.0))
    finally:
        await agen.aclose()
    assert [e.sequence for e in got] == [2, 3]


# -- live delivery ----------------------------------------------------------
async def test_a_live_subscriber_sees_a_newly_emitted_event():
    bus = _bus()
    agen = bus.subscribe("s")                       # from 0, nothing stored yet
    nxt = asyncio.ensure_future(agen.__anext__())
    try:
        await asyncio.sleep(0)                       # let it register and start waiting
        await bus.emit("s", EventType.RUN_FINISHED, {"ok": True})
        event = await asyncio.wait_for(nxt, 1.0)
        assert event.type == EventType.RUN_FINISHED
        assert event.data == {"ok": True}
    finally:
        if not nxt.done():
            nxt.cancel()
        await agen.aclose()


async def test_close_ends_open_subscriptions():
    bus = _bus()
    agen = bus.subscribe("s")
    nxt = asyncio.ensure_future(agen.__anext__())
    await asyncio.sleep(0)                           # subscriber is now waiting
    await bus.close()                                # asks every subscription to end

    ended = False
    try:
        await asyncio.wait_for(nxt, 1.0)
    except StopAsyncIteration:
        ended = True
    finally:
        await agen.aclose()
    assert ended

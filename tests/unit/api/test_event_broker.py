"""``EventBroker`` — delivery, late-subscriber replay, and clean teardown."""

from __future__ import annotations

import asyncio

from api.services.event_broker import EventBroker
from common.events import AgentEvent


async def _collect(broker: EventBroker, task_id: str) -> list[AgentEvent]:
    return [event async for event in broker.subscribe(task_id)]


async def _wait_subscribers(broker: EventBroker, task_id: str, count: int = 1) -> None:
    """Spin the loop until ``count`` subscribers have registered under the lock."""
    while True:
        topic = broker._topics.get(task_id)
        if topic is not None and len(topic.subscribers) >= count:
            return
        await asyncio.sleep(0)


async def test_late_subscriber_replays_buffered_events():
    broker = EventBroker()
    await broker.publish("t", AgentEvent("state", {"i": 0}))
    await broker.publish("t", AgentEvent("finished", {"i": 1}))
    await broker.close("t")

    # Subscribing after the run finished still yields the whole history.
    events = await _collect(broker, "t")
    assert [e.type for e in events] == ["state", "finished"]


async def test_live_delivery_in_order():
    broker = EventBroker()
    collector = asyncio.create_task(_collect(broker, "t"))
    await _wait_subscribers(broker, "t")

    await broker.publish("t", AgentEvent("a", {}))
    await broker.publish("t", AgentEvent("b", {}))
    await broker.close("t")

    events = await collector
    assert [e.type for e in events] == ["a", "b"]


async def test_close_sentinel_ends_generator():
    broker = EventBroker()
    collector = asyncio.create_task(_collect(broker, "t"))
    await _wait_subscribers(broker, "t")

    await broker.close("t")  # no events, just the terminal sentinel

    assert await collector == []


async def test_two_subscribers_both_see_every_event():
    broker = EventBroker()
    c1 = asyncio.create_task(_collect(broker, "t"))
    c2 = asyncio.create_task(_collect(broker, "t"))
    await _wait_subscribers(broker, "t", count=2)

    await broker.publish("t", AgentEvent("x", {}))
    await broker.close("t")

    r1, r2 = await asyncio.gather(c1, c2)
    assert [e.type for e in r1] == ["x"]
    assert [e.type for e in r2] == ["x"]


async def test_subscriber_unregistered_after_stream_ends():
    broker = EventBroker()
    collector = asyncio.create_task(_collect(broker, "t"))
    await _wait_subscribers(broker, "t")
    await broker.close("t")
    await collector

    # The generator's finally clause must have discarded its queue.
    assert broker._topics["t"].subscribers == set()


async def test_publish_after_close_is_ignored():
    broker = EventBroker()
    await broker.publish("t", AgentEvent("state", {}))
    await broker.close("t")
    await broker.publish("t", AgentEvent("late", {}))  # dropped: topic is closed

    events = await _collect(broker, "t")
    assert [e.type for e in events] == ["state"]

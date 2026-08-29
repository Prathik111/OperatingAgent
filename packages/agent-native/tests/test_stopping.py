"""Stopping a run, and what a stop is allowed to leave behind.

A stop that's only noticed between turns arrives after the thing the user wanted
stopped has already happened, so the loop checks for one between stream chunks and
between tool groups too. That's the behaviour here, along with the two rules that
keep a stopped conversation usable:

  * a half-streamed tool call is dropped, because half a JSON object isn't a
    smaller request - it's a broken one;
  * a call that was complete but never got to run still gets a result saying so,
    because the wire format requires a result for every call the assistant made.

Offline: the model is `_scripted.ScriptedProvider`, so the test decides exactly
which chunk the stop lands on.
"""

from __future__ import annotations

from typing import Any

from agent_native.config import AgentConfig
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    Role,
    Session,
    ToolCall,
    ToolCallStatus,
    assistant_message,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.loop import AgentLoop, Cancellation, Limits, RunContext, RunStatus
from agent_native.tools.base import ToolRegistry, ToolResult
from tests._scripted import ScriptedProvider, call_event, scripted_registry, text_event


class NoopToolManager:
    """Authorizes everything and runs nothing that matters.

    The point of these tests is which calls reach a tool at all, so `ran` is the
    only thing worth recording.
    """

    def __init__(self, on_run: Any = None) -> None:
        self.ran: list = []
        self._on_run = on_run

    async def authorize(self, call: Any, context: Any):
        return None  # None means "not refused"

    async def run_authorized(self, call: Any, context: Any) -> ToolResult:
        self.ran.append(call.name)
        if self._on_run is not None:
            self._on_run()
        return ToolResult(True, output="ok")


def _loop(provider: ScriptedProvider, tool_manager: Any = None):
    db = MemoryDatabase()
    return (
        AgentLoop(
            scripted_registry(provider),
            ToolRegistry(),
            tool_manager,
            ContextManager(),
            EventBus(db),
            db,
        ),
        db,
    )


def _context(session: Session, cancel: Cancellation, turns: int = 4) -> RunContext:
    return RunContext(
        session=session,
        run_id="run_test",
        config=AgentConfig(model="scripted-1"),
        limits=Limits(max_turns=turns),
        cancellation=cancel,
    )


async def test_a_stop_mid_stream_keeps_the_text_and_drops_the_tool_call():
    cancel = Cancellation()
    provider = ScriptedProvider(
        [
            text_event("I'll start by "),
            text_event("removing the old files"),
            call_event(0, "terminal_run_command", '{"command": "rm -rf build"}'),
        ]
    )
    # Stop once the first chunk has been handed over - so the tool call is still
    # in flight when the user says stop.
    seen = {"chunks": 0}

    def tick() -> None:
        seen["chunks"] += 1
        if seen["chunks"] == 2:
            cancel.cancel()

    provider.on_chunk = tick

    loop, db = _loop(provider)
    session = Session(agent="build")
    await db.create_session(session)
    conv = Conversation([system_message("sys"), user_message(session.id, "clean up the build")])

    result = await loop.run(conv, _context(session, cancel))

    assert result.status is RunStatus.CANCELLED
    assert "I'll start by" in result.final_text          # what reached the screen is kept
    assistant = [m for m in conv.messages if m.role == Role.ASSISTANT][-1]
    assert not assistant.has_tool_calls()                # `rm -rf build` never became a call
    assert conv.is_valid()                               # nothing left waiting for a result
    assert provider.closed                               # and the stream wasn't left open


async def test_a_stop_between_tool_groups_still_answers_every_call():
    """The first call runs, the user stops, the rest come back as 'not run'.

    Two writes are two groups (only reads share one), so the stop lands between
    them - and the conversation still holds a result for both, which is the only
    reason it can be sent to a model again.
    """
    cancel = Cancellation()
    manager = NoopToolManager(on_run=cancel.cancel)
    loop, db = _loop(ScriptedProvider([]), manager)
    session = Session(agent="build")
    await db.create_session(session)

    calls = [
        ToolCall(id="c1", name="filesystem_write_file", arguments={"path": "a.txt"}),
        ToolCall(id="c2", name="filesystem_write_file", arguments={"path": "b.txt"}),
    ]
    assistant = assistant_message(session.id, text="writing both", tool_calls=calls)
    conv = Conversation([system_message("sys"), assistant])

    await loop.run_tools(assistant, conv, _context(session, cancel))

    assert manager.ran == ["filesystem_write_file"]      # only the first one
    results = [c for m in conv.messages if m.role == Role.TOOL for c in m.tool_calls()]
    assert [c.id for c in results] == ["c1", "c2"]        # both answered, in order
    assert results[0].output == "ok"
    assert "you stopped the agent" in results[1].error
    assert results[1].status is ToolCallStatus.ERROR
    assert conv.is_valid()


async def test_a_run_records_the_front_of_its_request():
    """The fingerprint is written down even on a run where nothing moved -
    otherwise 'it never changed' and 'nobody looked' are the same answer."""
    cancel = Cancellation()
    loop, db = _loop(ScriptedProvider([text_event("done")]))
    session = Session(agent="build")
    await db.create_session(session)
    context = _context(session, cancel, turns=1)

    result = await loop.run(
        Conversation([system_message("sys"), user_message(session.id, "hi")]), context
    )

    assert result.status is RunStatus.FINISHED
    assert context.prefix_fingerprint          # something was recorded
    assert context.prefix_changed is False     # and it held steady


async def test_a_prefix_that_moves_is_noticed():
    """Called twice with a different front, the run is marked as having moved."""
    cancel = Cancellation()
    loop, _db = _loop(ScriptedProvider([]))
    context = _context(Session(agent="build"), cancel)

    loop._check_prefix([{"role": "system", "content": "steady"}], [], context)
    assert context.prefix_changed is False
    loop._check_prefix([{"role": "system", "content": "steady. Today is Tuesday"}], [], context)
    assert context.prefix_changed is True

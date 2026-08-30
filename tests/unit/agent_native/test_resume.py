"""Resuming a run from where the log ends, with no side effect done twice.

Every step of a run is written down as it happens - each message to the database,
each event to the numbered log - so a run that dies partway through (the process
killed, the machine rebooted, a deploy cycled the server) doesn't have to start
over. It can be picked up from exactly the state on disk. Step 11's promise is that
picking it up is *safe*: a tool call that already ran is never run again.

A crashed run leaves the stored conversation ending in one of a few shapes, and
these tests cover each one, at the loop and then through the service:

  * an assistant turn whose tool call never got a result -> the call is finished,
    exactly once, and the run carries on;
  * a call that already has a saved result -> it is left alone, never re-run - the
    no-duplicate-side-effects guarantee;
  * a batch where some calls ran and some didn't -> only the missing ones run;
  * a final answer that was saved but never recorded -> it's handed back without
    asking the model to think again;
  * through the service: a reattached run continues under the same run id, and a
    run whose log already ends in RUN_FINISHED is a no-op that still returns the
    receipt without touching the model.

Offline by construction: the model is `_scripted.ScriptedProvider`, no network and
no key. Run under pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src:packages/agent-native \
        python3 packages/agent-native/tests/test_resume.py
"""

from __future__ import annotations

import asyncio
import sys
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
    tool_result_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import Event, EventBus, EventType
from agent_native.loop import AgentLoop, Cancellation, Limits, RunContext, RunStatus
from agent_native.service import AgentRuntime, AgentService
from agent_native.tools.base import ToolRegistry, ToolResult
from tests._scripted import ScriptedProvider, scripted_registry, text_event


class CountingToolManager:
    """Authorizes everything and counts how many times each tool actually runs.

    The whole point of resume is *which* calls reach a tool, so the count per name
    is the only thing worth recording: a re-run would show up as a count of two.
    """

    def __init__(self) -> None:
        self.runs: dict = {}  # tool name -> times run

    async def authorize(self, call: Any, context: Any) -> Any:
        return None  # None means "not refused"

    async def run_authorized(self, call: Any, context: Any) -> ToolResult:
        self.runs[call.name] = self.runs.get(call.name, 0) + 1
        return ToolResult(True, output=f"{call.name} ran")


def _loop(provider: ScriptedProvider, tool_manager: Any = None) -> tuple:
    db = MemoryDatabase()
    loop = AgentLoop(
        scripted_registry(provider),
        ToolRegistry(),
        tool_manager or CountingToolManager(),
        ContextManager(),
        EventBus(db),
        db,
    )
    return loop, db


def _context(session: Session, run_id: str = "run_resume", turns: int = 6) -> RunContext:
    return RunContext(
        session=session,
        run_id=run_id,
        config=AgentConfig(model="scripted-1"),
        limits=Limits(max_turns=turns),
        cancellation=Cancellation(),
    )


def _tool_results(conv: Conversation) -> list:
    """Every finished tool call in the conversation, in order."""
    return [c for m in conv.messages if m.role == Role.TOOL for c in m.tool_calls()]


# ---------------------------------------------------------------------------
# At the loop: the conversation tail decides what resume does
# ---------------------------------------------------------------------------
async def test_a_pending_tool_call_is_finished_and_run_once() -> None:
    """A turn asked for a tool and died before the result was saved.

    Resume finishes that call - once - then lets the loop carry on to the answer.
    """
    manager = CountingToolManager()
    loop, db = _loop(ScriptedProvider([text_event("done")]), manager)
    session = Session(agent="build")
    await db.create_session(session)

    request = ToolCall(id="c0", name="read_thing", arguments={"path": "a.txt"})
    conv = Conversation(
        [
            system_message("sys"),
            user_message(session.id, "read it"),
            assistant_message(session.id, text="reading", tool_calls=[request]),
        ]
    )

    result = await loop.run(conv, _context(session))

    assert manager.runs == {"read_thing": 1}          # finished, and exactly once
    assert [c.id for c in _tool_results(conv)] == ["c0"]  # its result is now on record
    assert conv.is_valid()
    assert result.status is RunStatus.FINISHED
    assert result.final_text == "done"
    assert len(provider_requests(loop)) == 1          # one turn, to read the result


async def test_a_saved_result_is_never_rerun() -> None:
    """The call already has a result. Resume must not run it a second time."""
    manager = CountingToolManager()
    provider = ScriptedProvider([text_event("done")])
    loop, db = _loop(provider, manager)
    session = Session(agent="build")
    await db.create_session(session)

    request = ToolCall(id="c0", name="delete_everything", arguments={})
    done = ToolCall(id="c0", name="delete_everything", status=ToolCallStatus.SUCCESS, output="ok")
    conv = Conversation(
        [
            system_message("sys"),
            user_message(session.id, "go"),
            assistant_message(session.id, text="deleting", tool_calls=[request]),
            tool_result_message(session.id, done),
        ]
    )

    result = await loop.run(conv, _context(session))

    assert manager.runs == {}                          # the dangerous call never re-ran
    assert result.status is RunStatus.FINISHED
    assert result.final_text == "done"
    assert len(provider.requests) == 1                 # just the turn that reads the result


async def test_a_partial_batch_runs_only_the_missing_call() -> None:
    """Two calls were asked for, one finished before the crash. Only the other runs."""
    manager = CountingToolManager()
    loop, db = _loop(ScriptedProvider([text_event("done")]), manager)
    session = Session(agent="build")
    await db.create_session(session)

    first = ToolCall(id="c1", name="read_alpha", arguments={})
    second = ToolCall(id="c2", name="read_beta", arguments={})
    first_done = ToolCall(id="c1", name="read_alpha", status=ToolCallStatus.SUCCESS, output="alpha")
    conv = Conversation(
        [
            system_message("sys"),
            user_message(session.id, "read both"),
            assistant_message(session.id, text="reading both", tool_calls=[first, second]),
            tool_result_message(session.id, first_done),
        ]
    )

    result = await loop.run(conv, _context(session))

    assert manager.runs == {"read_beta": 1}            # only the one that never ran
    assert [c.id for c in _tool_results(conv)] == ["c1", "c2"]  # both answered now
    assert conv.is_valid()
    assert result.status is RunStatus.FINISHED


async def test_a_saved_final_answer_is_not_regenerated() -> None:
    """The model's final answer was saved but the run never recorded itself.

    Resume must hand that answer back, not pay for a fresh turn - so a provider that
    would say something else is never asked, and the run still finishes and records.
    """
    provider = ScriptedProvider([text_event("REGENERATED - should not appear")])
    loop, db = _loop(provider)
    session = Session(agent="build")
    await db.create_session(session)
    conv = Conversation(
        [
            system_message("sys"),
            user_message(session.id, "answer me"),
            assistant_message(session.id, text="the final answer"),
        ]
    )

    result = await loop.run(conv, _context(session, run_id="run_kept"))

    assert provider.requests == []                     # the model was never asked
    assert result.status is RunStatus.FINISHED
    assert result.final_text == "the final answer"     # the saved answer, verbatim
    # And the run recorded itself on the way out, so it now looks finished to a resume.
    events = await db.load_events(session.id, 0)
    assert events and events[-1].type == EventType.RUN_FINISHED
    assert events[-1].run_id == "run_kept"


# ---------------------------------------------------------------------------
# Through the service: reattach, and the finished-run no-op
# ---------------------------------------------------------------------------
async def test_resume_reattaches_and_finishes_the_run() -> None:
    """A crashed run, all messages on disk, is carried to a finish under its own id."""
    db = MemoryDatabase()
    provider = ScriptedProvider([text_event("all wrapped up")])
    runtime = AgentRuntime(
        database=db,
        model_registry=scripted_registry(provider),
        agents=[AgentConfig(name="build", model="scripted-1")],
    )
    service = AgentService(runtime)

    session = Session(agent="build")
    await db.create_session(session)
    await db.save_message(system_message("sys", session.id))
    await db.save_message(user_message(session.id, "do the thing"))
    request = ToolCall(id="c0", name="read_thing", arguments={})
    done = ToolCall(id="c0", name="read_thing", status=ToolCallStatus.SUCCESS, output="ok")
    await db.save_message(assistant_message(session.id, text="reading", tool_calls=[request]))
    await db.save_message(tool_result_message(session.id, done))
    # One event from the interrupted run, so resume knows which run to continue.
    seq = await db.next_sequence(session.id)
    await db.save_event(
        Event(sequence=seq, type=EventType.TURN_STARTED, session_id=session.id,
              run_id="run_orig", data={"turn": 1})
    )

    result = await service.resume_run(session.id)

    assert result.status is RunStatus.FINISHED
    assert result.final_text == "all wrapped up"
    assert result.run_id == "run_orig"                 # same run, not a fork
    assert provider.requests                           # the model was asked to finish
    # The finish was recorded under that same id.
    saved = await db.get_run("run_orig")
    assert saved is not None and saved.status == "finished"
    # It continued *from the last event*: the seeded event stays first, the run's
    # new events are appended after that cursor (higher sequence) under the same
    # run id, and the log now ends in RUN_FINISHED.
    events = await db.load_events(session.id, 0)
    assert events[0].sequence == seq and events[0].type == EventType.TURN_STARTED
    appended = [e for e in events if e.sequence > seq]
    assert appended, "resume should append new events past the cursor"
    assert all(e.run_id == "run_orig" for e in appended)
    assert events[-1].type == EventType.RUN_FINISHED


async def test_resume_of_a_finished_run_does_not_call_the_model() -> None:
    """The log already ends in RUN_FINISHED: resume rebuilds the receipt, no model call."""
    db = MemoryDatabase()
    provider = ScriptedProvider([text_event("SHOULD NOT BE CALLED")])
    runtime = AgentRuntime(
        database=db,
        model_registry=scripted_registry(provider),
        agents=[AgentConfig(name="build", model="scripted-1")],
    )
    service = AgentService(runtime)

    session = Session(agent="build")
    await db.create_session(session)
    await db.save_message(system_message("sys", session.id))
    await db.save_message(user_message(session.id, "already answered"))
    await db.save_message(assistant_message(session.id, text="the final answer"))
    seq = await db.next_sequence(session.id)
    await db.save_event(
        Event(
            sequence=seq, type=EventType.RUN_FINISHED, session_id=session.id, run_id="run_done",
            data={
                "run_id": "run_done", "status": "finished", "turns": 2,
                "final_text": "the final answer", "input_tokens": 30, "output_tokens": 12,
                "cached_tokens": 0, "duration_seconds": 1.5, "cost_usd": 0.0,
                "model": "scripted-1", "retries": 0, "stop_reason": "",
            },
        )
    )

    result = await service.resume_run(session.id)

    assert provider.requests == []                     # the model was never asked
    assert result.run_id == "run_done"
    assert result.status is RunStatus.FINISHED
    assert result.final_text == "the final answer"
    assert result.turns == 2
    assert result.usage.input_tokens == 30 and result.usage.output_tokens == 12
    assert result.duration_seconds == 1.5


def provider_requests(loop: AgentLoop) -> list:
    """The scripted provider behind a loop, for asserting how often it was asked."""
    return loop._models.get_provider(loop._models.get("scripted-1")).requests


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_a_pending_tool_call_is_finished_and_run_once,
        test_a_saved_result_is_never_rerun,
        test_a_partial_batch_runs_only_the_missing_call,
        test_a_saved_final_answer_is_not_regenerated,
        test_resume_reattaches_and_finishes_the_run,
        test_resume_of_a_finished_run_does_not_call_the_model,
    ]
    failures: list = []
    for test in tests:
        try:
            asyncio.run(test())
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - resume:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - resume: {len(tests)} tests "
          "(pending call finished once, saved result never re-run, partial batch, "
          "saved answer kept, service reattach + finished-run no-op).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

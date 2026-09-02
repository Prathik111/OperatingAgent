"""Parallel subagent fan-out: one helper mapped over a list of jobs and run at
once, then gathered into one labelled answer.

Step 26's promise isn't raw concurrency - `DelegateTool` could already be called
in a loop - but making the parallel case first-class and safe. The plan's own
verify names four things, and each has a test here that would fail if it regressed:

  * **The turn costs about the slowest child, not the sum.** Five jobs that each
    sleep 0.15s finish in about 0.15s, not 0.75s, and the high-water mark of
    overlapping model calls reaches the full width - the direct, non-timing proof
    that they really ran together (`test_fan_out_runs_children_concurrently`).
  * **Each child's stream stays attributable.** Every child is an ordinary helper
    run with a `#i` suffix on its run id, so its receipt row and its events on the
    parent's stream are traceable to that one job (`test_each_child_is_attributable`).
  * **One cancel stops all of them.** Cancelling the parent stops every child at its
    next turn boundary; pre-cancelled, not one reaches a model
    (`test_cancelling_the_parent_stops_every_child`).
  * **It can't multiply into a runaway bill.** Each child honours the per-child turn
    cap, so a helper that loops forever costs `cap x width` turns, not the parent's
    whole budget times the width (`test_per_child_turn_cap_prevents_a_runaway`).

The rest guard the edges: bad `jobs` are refused with a fixable message rather than
run, a plain `delegate` call's run id is byte-for-byte what it always was (the
`#i` suffix is fan-out-only), and a helper is handed neither delegation tool so it
can't open a wave of its own.

Offline by construction: scripted stand-in models, no network, no key. Run under
pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src:packages/agent-native \
        python3 packages/agent-native/tests/test_fanout.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from agent_native.config import AgentConfig, Subagent
from agent_native.conversation import Session
from agent_native.database import MemoryDatabase
from agent_native.loop import Cancellation, Limits, RunContext
from agent_native.models.base import Model, ModelRegistry, StreamEvent, StreamType
from agent_native.service import AgentRuntime
from agent_native.tools.base import (
    Tool,
    ToolDefinition,
    ToolPermissions,
    ToolRegistry,
    ToolResult,
)
from agent_native.tools.subagent import (
    FANOUT_TOOL_NAME,
    HELPER_RUN_SEPARATOR,
    MAX_FANOUT_WIDTH,
    TOOL_NAME,
    DelegateTool,
    FanOutTool,
)

from tests._scripted import call_event, text_event


def _usage_event(input_tokens: int, output_tokens: int) -> StreamEvent:
    return StreamEvent(
        StreamType.USAGE, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


def _last_user_text(messages: list) -> str:
    """The text of the last user message on the rendered wire.

    A child's conversation is `[system, user(job)]`, so this is how a stand-in
    provider reads back the job it was handed - and how the test knows each child
    got its *own* job and not another child's.
    """
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multimodal shape, defensive
                return " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
    return ""


# ---------------------------------------------------------------------------
# Stand-in providers whose timing and behaviour the test owns
# ---------------------------------------------------------------------------
class _SleepyAnswer:
    """A helper's model that sleeps, then answers with its own job echoed back.

    The sleep is what makes concurrency observable. `max_concurrent` records the
    high-water mark of overlapping calls: it can only reach the fan-out's full width
    if every child was inside `stream` at the same moment, which is a direct proof
    of parallelism that doesn't depend on wall-clock thresholds. The counters are
    bumped synchronously before the first `await`, so a child that has begun is
    counted before it can yield the loop to another.
    """

    def __init__(self, delay: float = 0.15) -> None:
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0
        self._delay = delay

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        job = _last_user_text(messages)
        try:
            await asyncio.sleep(self._delay)
            yield _usage_event(5, 5)
            yield text_event(f"done: {job}")
        finally:
            self._active -= 1

    def count_tokens(self, messages: list) -> int:
        return 0


class _LoopsForever:
    """A helper's model that calls a tool every turn and never answers.

    Left unchecked it would run until some ceiling stopped it - which is the point:
    it proves the per-child turn cap is the ceiling that fires, not the parent's own
    turn budget. Shared across children, so `calls` is the total across the wave.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        yield call_event(0, "noop", "{}")

    def count_tokens(self, messages: list) -> int:
        return 0


class _NoopTool(Tool):
    """A read-only tool that does nothing - just something for a looping helper to
    call so its turns actually advance. Read-only, so the default policy allows it
    without a prompt."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="noop",
            description="Does nothing.",
            input_schema={"type": "object", "properties": {}},
            permissions=ToolPermissions(read_only=True),
        )

    def preview(self, arguments: dict) -> str:
        return "noop"

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        return ToolResult(True, output="ok")


# ---------------------------------------------------------------------------
# Wiring: a runtime whose only helper is one worker on a scripted model
# ---------------------------------------------------------------------------
def _model(name: str, provider: str) -> Model:
    return Model(provider=provider, model_id=name, context_size=100_000)


def _runtime_with_worker(provider: Any, register_noop: bool = False) -> tuple:
    """A runtime with a single 'worker' subagent bound to `provider`.

    The parent shares the worker's model - it never runs here anyway; the tests
    drive the fan-out tool directly - so one registered provider serves everything.
    """
    reg = ModelRegistry()
    reg.register_provider("worker", provider)
    reg.register_model("worker-1", _model("worker-1", "worker"))

    tools = ToolRegistry()
    if register_noop:
        tools.register(_NoopTool())

    db = MemoryDatabase()
    parent = AgentConfig(
        name="build",
        model="worker-1",
        subagents=[Subagent(name="worker", model="worker-1")],
    )
    runtime = AgentRuntime(
        database=db, model_registry=reg, tool_registry=tools, agents=[parent]
    )
    return runtime, db


def _context(
    runtime: AgentRuntime,
    session: Session,
    cancellation: Cancellation | None = None,
    helper_max_turns: int = 0,
) -> RunContext:
    # max_parallel_tools=8 so a width-5 fan-out isn't throttled below its width -
    # the concurrency test needs all five able to run at once. helper_max_turns
    # defaults to 0 (use the tool's own cap); a test sets it to pin the runaway guard.
    return RunContext(
        session=session,
        run_id="run_parent",
        config=runtime.config_for("build"),
        limits=Limits(
            max_turns=5, max_retries=0, max_parallel_tools=8, helper_max_turns=helper_max_turns
        ),
        cancellation=cancellation or Cancellation(),
    )


# ---------------------------------------------------------------------------
# The plan's verify, in four parts
# ---------------------------------------------------------------------------
async def test_fan_out_runs_children_concurrently() -> None:
    """Five jobs run at once: the wall clock is about one child, not five, and the
    overlap high-water mark reaches the full width."""
    worker = _SleepyAnswer(delay=0.15)
    runtime, db = _runtime_with_worker(worker)
    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    context = _context(runtime, session)

    jobs = [f"job {i}" for i in range(5)]
    start = time.perf_counter()
    result = await FanOutTool(runtime).execute({"helper": "worker", "jobs": jobs}, context)
    elapsed = time.perf_counter() - start

    assert result.success is True
    assert worker.calls == 5                     # every job reached the model once
    assert worker.max_concurrent == 5            # ...and all five overlapped
    assert elapsed < 0.6                          # about one 0.15s child, not 5 x 0.15 = 0.75
    for job in jobs:
        assert f"done: {job}" in result.output   # each child answered its own job


async def test_each_child_is_attributable() -> None:
    """Every child has its own receipt row and its own run id on the parent's event
    stream, each tagged with the `#i` that ties it to one job."""
    worker = _SleepyAnswer(delay=0.0)
    runtime, db = _runtime_with_worker(worker)
    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    context = _context(runtime, session)

    jobs = [f"job {i}" for i in range(5)]
    result = await FanOutTool(runtime).execute({"helper": "worker", "jobs": jobs}, context)
    assert result.success is True

    # A receipt per child, keyed by the parent run id plus /worker#i.
    for index in range(5):
        run_id = f"run_parent{HELPER_RUN_SEPARATOR}worker#{index}"
        assert await db.get_run(run_id) is not None, f"no receipt for {run_id}"

    # ...and those same run ids appear on the parent session's events, so a watcher
    # can attribute each event to the child that emitted it.
    events = await db.load_events(session.id, 0)
    seen = {getattr(e, "run_id", "") for e in events}
    for index in range(5):
        assert f"run_parent{HELPER_RUN_SEPARATOR}worker#{index}" in seen


async def test_cancelling_the_parent_stops_every_child() -> None:
    """A pre-cancelled parent stops the whole wave: the call fails cleanly and not
    one child ever reaches the model."""
    worker = _SleepyAnswer(delay=0.15)
    runtime, db = _runtime_with_worker(worker)
    session = Session(agent="build", working_directory=".")
    await db.create_session(session)

    cancel = Cancellation()
    cancel.cancel()  # every child checks this at the top of its first turn
    context = _context(runtime, session, cancellation=cancel)

    jobs = [f"job {i}" for i in range(5)]
    result = await FanOutTool(runtime).execute({"helper": "worker", "jobs": jobs}, context)

    assert result.success is False
    assert "cancel" in result.error.lower()
    assert worker.calls == 0                      # stopped before any provider call


async def test_per_child_turn_cap_prevents_a_runaway() -> None:
    """A helper that loops forever is stopped by the per-child turn cap, not the
    parent's budget: three children at a cap of two cost six turns, and each comes
    back marked out-of-turns rather than crashing the whole fan-out."""
    worker = _LoopsForever()
    runtime, db = _runtime_with_worker(worker, register_noop=True)
    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    context = _context(runtime, session, helper_max_turns=2)

    jobs = ["a", "b", "c"]
    result = await FanOutTool(runtime).execute({"helper": "worker", "jobs": jobs}, context)

    assert result.success is True                 # limit-reached is partial, still worth returning
    assert worker.calls == 6                      # 2 turns x 3 children, not the parent's 5
    assert result.output.count("ran out of turns") == 3


# ---------------------------------------------------------------------------
# The edges: validation, the unchanged single-delegate path, and anti-recursion
# ---------------------------------------------------------------------------
async def test_validation_rejects_bad_jobs() -> None:
    """Bad `jobs` are refused with a message that says how to fix it, and nothing
    runs. An over-wide wave is refused whole, never silently truncated."""
    worker = _SleepyAnswer(delay=0.0)
    runtime, db = _runtime_with_worker(worker)
    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    context = _context(runtime, session)
    tool = FanOutTool(runtime)

    empty = await tool.execute({"helper": "worker", "jobs": []}, context)
    assert empty.success is False and "empty" in empty.error.lower()

    not_a_list = await tool.execute({"helper": "worker", "jobs": "just one"}, context)
    assert not_a_list.success is False and "list" in not_a_list.error.lower()

    too_wide = await tool.execute(
        {"helper": "worker", "jobs": [f"j{i}" for i in range(MAX_FANOUT_WIDTH + 1)]}, context
    )
    assert too_wide.success is False and "too many" in too_wide.error.lower()

    unknown = await tool.execute({"helper": "ghost", "jobs": ["do it"]}, context)
    assert unknown.success is False and "no helper" in unknown.error.lower()

    assert worker.calls == 0                      # not one refusal ran a child


async def test_delegate_run_id_is_unchanged_without_a_suffix() -> None:
    """A plain `delegate` call's receipt is byte-for-byte what it always was:
    `{parent}/{name}` with no `#suffix`. The suffix is fan-out's alone."""
    worker = _SleepyAnswer(delay=0.0)
    runtime, db = _runtime_with_worker(worker)
    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    context = _context(runtime, session)

    result = await DelegateTool(runtime).execute({"helper": "worker", "job": "do it"}, context)
    assert result.success is True
    assert await db.get_run(f"run_parent{HELPER_RUN_SEPARATOR}worker") is not None
    assert await db.get_run(f"run_parent{HELPER_RUN_SEPARATOR}worker#0") is None


async def test_a_helper_cannot_fan_out_or_delegate() -> None:
    """A helper's tool list has both delegation tools removed, so no helper hires a
    helper and no helper opens a fan-out of its own - while ordinary tools pass
    straight through."""
    worker = _SleepyAnswer(delay=0.0)
    runtime, _ = _runtime_with_worker(worker, register_noop=True)

    allowed = FanOutTool(runtime)._tools_for_helper(AgentConfig())
    assert TOOL_NAME not in allowed               # "delegate"
    assert FANOUT_TOOL_NAME not in allowed        # "fan_out"
    assert "noop" in allowed                       # an ordinary tool is untouched


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_fan_out_runs_children_concurrently,
        test_each_child_is_attributable,
        test_cancelling_the_parent_stops_every_child,
        test_per_child_turn_cap_prevents_a_runaway,
        test_validation_rejects_bad_jobs,
        test_delegate_run_id_is_unchanged_without_a_suffix,
        test_a_helper_cannot_fan_out_or_delegate,
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
        print("FAIL - fanout:")
        for line in failures:
            print("  -", line)
        return 1
    print(
        f"PASS - fanout: {len(tests)} tests (concurrency + overlap, attribution, "
        "shared cancel, per-child turn cap, validation, unchanged delegate id, "
        "anti-recursion)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

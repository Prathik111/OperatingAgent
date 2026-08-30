"""Lifecycle hooks: user callbacks at defined moments in a run.

Step 16 adds five points a user can attach a callback to - before a tool runs,
after it returns, when a prompt is submitted, and when a run or a subagent stops -
so someone can bolt on auto-formatting, logging, or an extra policy gate without
editing the agent. The event bus already announces these same moments; hooks earn
their place by running *in-band* (so a slow hook actually holds the step) and by
letting a pre-tool hook **veto** a call.

These tests check the three promises the plan's verify names, and the wiring that
makes them true everywhere:

  * a pre-tool hook that blocks a command is honored - the tool never runs, and
    the model reads an ordinary refusal it can react to (it is a hard stop, not a
    prompt);
  * a post-tool hook fires with the result of a call that actually ran;
  * disabling hooks restores current behaviour exactly - an empty manager is a
    no-op at every point, guarded so the old code path is taken unchanged.

Plus the two points the tool path doesn't cover: prompt-submitted fires from the
service when a message is accepted (and, being observe-only, cannot veto the run),
and the stop point fires from the loop's one finish chokepoint - RUN_STOP for a
top-level conversation, SUBAGENT_STOP for a helper, told apart by the helper
separator in the run id, because a helper never reaches the service.

Offline by construction: a scripted stand-in model, no network, no key. Run under
pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src:packages/agent-native python3 packages/agent-native/tests/test_hooks.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
from typing import Any

from agent_native.config import AgentConfig
from agent_native.context import ContextManager
from agent_native.conversation import Conversation, Session, system_message, user_message
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.hooks import HookContext, HookManager, HookOutcome, HookPoint
from agent_native.loop import (
    AgentLoop,
    Cancellation,
    Limits,
    RunContext,
    RunStatus,
)
from agent_native.models.base import ModelRegistry
from agent_native.permissions import Decision, PermissionDecision, Policy, PolicyChain
from agent_native.service import AgentRuntime, AgentService
from agent_native.tools.base import ToolRegistry
from agent_native.tools.manager import ToolManager

from tests._fake_tools import ReadFileTool, WriteFileTool
from tests._scripted import (
    ScriptedProvider,
    call_event,
    scripted_model,
    scripted_registry,
    text_event,
)


# ---------------------------------------------------------------------------
# Tiny doubles
# ---------------------------------------------------------------------------
class TurnAwareProvider:
    """Turn one asks to write a file; every turn after finishes with text.

    A plain scripted provider replays the *same* events each turn, which here would
    loop forever re-asking to write (or re-asking after a veto). The model has to
    change what it says between turns for a run to end, so this stands in for that.
    """

    def __init__(self, write_args: str) -> None:
        self._write_args = write_args
        self.requests: list = []
        self.calls = 0
        self.closed = False

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.requests.append((messages, tools))
        self.calls += 1
        try:
            if self.calls == 1:
                yield call_event(0, "write_file", self._write_args)
            else:
                yield text_event("done")
        finally:
            self.closed = True

    def count_tokens(self, messages: list) -> int:
        return 0


class _AllowAll(Policy):
    """Allows everything, so a test isolates the *hook's* effect from the policy's.

    The write tool is destructive and would otherwise be asked about; with this in
    the chain the only thing that can stop a write is a pre-tool hook - which is
    exactly what the veto test needs to prove.
    """

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        return Decision(PermissionDecision.ALLOW, reason="allow-all")


class _MustNotAsk:
    """A prompter that fails if used: a hook veto is a hard stop, never a prompt."""

    def __init__(self) -> None:
        self.asked = 0

    async def ask(self, request: Any, session_id: str) -> bool:
        self.asked += 1
        raise AssertionError("a hook veto must refuse outright, never prompt the user")


def _blocker(reason: str):
    """A pre-tool hook that vetoes every call it sees."""

    def hook(ctx: HookContext) -> HookOutcome:
        return HookOutcome(block=True, reason=reason)

    return hook


def _recorder(sink: list):
    """An observe-only hook that appends the context it was handed."""

    def hook(ctx: HookContext) -> None:
        sink.append(ctx)

    return hook


# ---------------------------------------------------------------------------
# HookManager, in isolation
# ---------------------------------------------------------------------------
async def test_empty_manager_is_a_noop() -> None:
    manager = HookManager()
    for point in HookPoint:
        assert manager.has(point) is False
        # Dispatch on an empty point returns immediately with no veto - the property
        # every call site leans on so "no hooks registered" == "old behaviour".
        assert await manager.dispatch(HookContext(point=point)) is None


async def test_hooks_run_in_registration_order() -> None:
    order: list = []
    manager = HookManager()
    manager.register(HookPoint.POST_TOOL, lambda c: order.append("first"))
    manager.register(HookPoint.POST_TOOL, lambda c: order.append("second"))
    await manager.dispatch(HookContext(point=HookPoint.POST_TOOL))
    assert order == ["first", "second"]


async def test_pre_tool_veto_is_returned() -> None:
    manager = HookManager()
    manager.register(HookPoint.PRE_TOOL, _blocker("nope"))
    outcome = await manager.dispatch(HookContext(point=HookPoint.PRE_TOOL))
    assert outcome is not None and outcome.block is True
    assert outcome.reason == "nope"


async def test_first_veto_wins_and_later_hooks_still_run() -> None:
    # A blocking hook doesn't cut the chain: an observer (a logger, say) registered
    # after it still gets to see the call. The *returned* veto is the first one.
    seen: list = []
    manager = HookManager()
    manager.register(HookPoint.PRE_TOOL, _blocker("first"))
    manager.register(HookPoint.PRE_TOOL, _recorder(seen))
    manager.register(HookPoint.PRE_TOOL, _blocker("second"))
    outcome = await manager.dispatch(HookContext(point=HookPoint.PRE_TOOL))
    assert outcome is not None and outcome.reason == "first"
    assert len(seen) == 1


async def test_a_raising_hook_is_isolated() -> None:
    # A hook that raises must not crash the run: the error is recorded and the next
    # hook still runs. This is what makes a careless logging callback harmless.
    ran: list = []

    def boom(ctx: HookContext) -> None:
        raise RuntimeError("kaboom")

    manager = HookManager()
    manager.register(HookPoint.POST_TOOL, boom)
    manager.register(HookPoint.POST_TOOL, lambda c: ran.append(True))
    ctx = HookContext(point=HookPoint.POST_TOOL)
    assert await manager.dispatch(ctx) is None
    assert ran == [True]
    assert len(ctx.errors) == 1 and "kaboom" in ctx.errors[0]


async def test_sync_and_async_hooks_both_supported() -> None:
    calls: list = []

    def sync_hook(ctx: HookContext) -> None:
        calls.append("sync")

    async def async_hook(ctx: HookContext) -> None:
        calls.append("async")

    manager = HookManager()
    manager.register(HookPoint.RUN_STOP, sync_hook)
    manager.register(HookPoint.RUN_STOP, async_hook)
    await manager.dispatch(HookContext(point=HookPoint.RUN_STOP))
    assert calls == ["sync", "async"]


async def test_clear_restores_the_noop() -> None:
    manager = HookManager()
    manager.register(HookPoint.PRE_TOOL, _blocker("x"))
    manager.register(HookPoint.POST_TOOL, _recorder([]))
    assert manager.has(HookPoint.PRE_TOOL) and manager.has(HookPoint.POST_TOOL)
    manager.clear(HookPoint.PRE_TOOL)               # one point
    assert not manager.has(HookPoint.PRE_TOOL)
    assert manager.has(HookPoint.POST_TOOL)
    manager.clear()                                 # all of them
    assert not manager.has(HookPoint.POST_TOOL)


# ---------------------------------------------------------------------------
# In the loop: the tool points (the plan's own verify)
# ---------------------------------------------------------------------------
def _loop(hooks: HookManager, provider: Any = None) -> tuple:
    """A real loop wired to the read/write fake tools and an allow-all policy chain.

    Allow-all so the only thing that can stop a write is a hook, and a must-not-ask
    prompter so a veto is proven to be a hard stop rather than a prompt the test
    happened to answer.
    """
    db = MemoryDatabase()
    provider = provider or TurnAwareProvider('{"path": "out.txt", "content": "hello"}')
    registry = ModelRegistry()
    registry.register_provider("scripted", provider)
    registry.register_model("scripted-1", scripted_model())
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    prompter = _MustNotAsk()
    manager = ToolManager(tools, PolicyChain([_AllowAll()]), prompter)
    loop = AgentLoop(registry, tools, manager, ContextManager(), EventBus(db), db, hooks=hooks)
    return loop, db, provider, prompter


def _context(session: Session) -> RunContext:
    return RunContext(
        session=session,
        run_id="run_hooks",                          # no separator: a top-level run
        config=AgentConfig(model="scripted-1"),
        limits=Limits(max_turns=5),
        cancellation=Cancellation(),
    )


async def _drive_write(hooks: HookManager) -> tuple:
    """Run a full conversation whose model tries to write a file. Report the outcome."""
    loop, db, _provider, prompter = _loop(hooks)
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(agent="build", working_directory=tmp)
        await db.create_session(session)
        conv = Conversation([system_message("sys"), user_message(session.id, "make a file")])
        result = await loop.run(conv, _context(session))
        wrote = os.path.isfile(os.path.join(tmp, "out.txt"))
    return result, wrote, prompter.asked


async def test_pre_tool_hook_blocks_the_command() -> None:
    # Register a blocker *and* a post-tool recorder on the same manager, so we can
    # prove both halves at once: the write is stopped, and because it never ran, the
    # post-tool hook never fires.
    post_seen: list = []
    hooks = HookManager()
    hooks.register(HookPoint.PRE_TOOL, _blocker("policy says no writes today"))
    hooks.register(HookPoint.POST_TOOL, _recorder(post_seen))

    result, wrote, asked = await _drive_write(hooks)

    assert result.status is RunStatus.FINISHED       # it read the refusal and answered
    assert wrote is False                            # the tool never ran
    assert asked == 0                                # a hard stop, not a prompt
    assert post_seen == []                           # post-tool doesn't fire for a vetoed call


async def test_post_tool_hook_fires_with_the_result() -> None:
    post_seen: list = []
    hooks = HookManager()
    hooks.register(HookPoint.POST_TOOL, _recorder(post_seen))

    result, wrote, _asked = await _drive_write(hooks)

    assert result.status is RunStatus.FINISHED
    assert wrote is True                             # allowed, so it ran
    # The hook fired once, for the write, and was handed the actual ToolResult.
    assert len(post_seen) == 1
    ctx = post_seen[0]
    assert ctx.point is HookPoint.POST_TOOL
    assert ctx.tool_name == "write_file"
    assert ctx.result is not None and ctx.result.success is True
    assert "Wrote" in (ctx.result.output or "")
    assert ctx.run_id == "run_hooks"
    # Step 18 threads two more fields to the tool points: write_file is a mutation
    # (not read-only), and the run's folder is carried through - which is what lets
    # the auto-checkpointer tell an edit from a read and know where to snapshot.
    assert ctx.read_only is False
    assert ctx.working_directory != ""


async def test_no_hooks_restores_current_behavior_exactly() -> None:
    # An empty manager: the same run goes through untouched - the write runs and the
    # file is written, exactly as a loop built before hooks existed would do.
    result, wrote, asked = await _drive_write(HookManager())
    assert result.status is RunStatus.FINISHED
    assert wrote is True
    assert asked == 0


# ---------------------------------------------------------------------------
# In the loop: the stop point (top-level vs helper routing)
# ---------------------------------------------------------------------------
async def test_run_stop_fires_at_the_end_of_a_top_level_run() -> None:
    seen: list = []
    hooks = HookManager()
    hooks.register(HookPoint.RUN_STOP, _recorder(seen))
    # A text-only provider, so the run finishes in one turn with no tool calls.
    loop, db, _provider, _prompter = _loop(hooks, provider=ScriptedProvider([text_event("all done")]))
    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    conv = Conversation([system_message("sys"), user_message(session.id, "say something")])
    result = await loop.run(conv, _context(session))

    assert result.status is RunStatus.FINISHED
    assert len(seen) == 1
    ctx = seen[0]
    assert ctx.point is HookPoint.RUN_STOP
    assert ctx.run_id == "run_hooks"
    assert ctx.text == "all done"                    # the run's final answer
    assert ctx.status == "finished"                  # RunStatus value, as a string


async def test_run_stop_routes_helper_ids_to_subagent_stop() -> None:
    # `_run_stop` is the loop's one finish chokepoint; it picks the point by the run
    # id. A helper's id carries the separator, so it fires SUBAGENT_STOP, not
    # RUN_STOP - the same distinction the event stream draws with `is_helper_run`.
    run_seen: list = []
    sub_seen: list = []
    hooks = HookManager()
    hooks.register(HookPoint.RUN_STOP, _recorder(run_seen))
    hooks.register(HookPoint.SUBAGENT_STOP, _recorder(sub_seen))
    loop, _db, _provider, _prompter = _loop(hooks)

    session = types.SimpleNamespace(id="s1")
    result = types.SimpleNamespace(
        final_text="hi", status=types.SimpleNamespace(value="finished")
    )
    await loop._run_stop(session, "run_top", result)              # top-level
    await loop._run_stop(session, "run_top/researcher", result)   # a helper

    assert [c.run_id for c in run_seen] == ["run_top"]
    assert [c.run_id for c in sub_seen] == ["run_top/researcher"]


# ---------------------------------------------------------------------------
# In the service: the prompt-submitted point, and the runtime wiring
# ---------------------------------------------------------------------------
def _service(db: MemoryDatabase, provider: ScriptedProvider) -> tuple:
    """A real AgentRuntime + AgentService on a scripted model - no key, no network."""
    runtime = AgentRuntime(
        database=db,
        model_registry=scripted_registry(provider),
        agents=[AgentConfig(name="build", model="scripted-1")],
    )
    return runtime, AgentService(runtime)


async def test_runtime_shares_one_hook_manager_with_its_loop() -> None:
    # A hook the user registers on the runtime must reach the loop that runs tools,
    # so the two have to be the *same* manager, not two copies.
    runtime, _service_obj = _service(MemoryDatabase(), ScriptedProvider([text_event("ok")]))
    assert isinstance(runtime.hooks, HookManager)
    assert runtime.loop._hooks is runtime.hooks


async def test_prompt_submitted_fires_on_send_message() -> None:
    seen: list = []
    db = MemoryDatabase()
    runtime, service = _service(db, ScriptedProvider([text_event("ok")]))
    runtime.hooks.register(HookPoint.PROMPT_SUBMITTED, _recorder(seen))

    session = await service.create_session(agent="build", title="t", working_directory=".")
    result = await service.send_message(session.id, "hello there")

    assert result.status is RunStatus.FINISHED
    assert len(seen) == 1
    ctx = seen[0]
    assert ctx.point is HookPoint.PROMPT_SUBMITTED
    assert ctx.text == "hello there"                 # the prompt, verbatim
    assert ctx.session_id == session.id
    assert ctx.run_id.startswith("run_")             # named the run it kicked off


async def test_prompt_submitted_cannot_veto_the_run() -> None:
    # Prompt-submitted is observe-only: only a pre-tool hook may veto. A block
    # outcome here is ignored, and the run proceeds as normal.
    db = MemoryDatabase()
    runtime, service = _service(db, ScriptedProvider([text_event("ok")]))
    runtime.hooks.register(HookPoint.PROMPT_SUBMITTED, _blocker("try to stop the run"))

    session = await service.create_session(agent="build", title="t", working_directory=".")
    result = await service.send_message(session.id, "run anyway")
    assert result.status is RunStatus.FINISHED


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_empty_manager_is_a_noop,
        test_hooks_run_in_registration_order,
        test_pre_tool_veto_is_returned,
        test_first_veto_wins_and_later_hooks_still_run,
        test_a_raising_hook_is_isolated,
        test_sync_and_async_hooks_both_supported,
        test_clear_restores_the_noop,
        test_pre_tool_hook_blocks_the_command,
        test_post_tool_hook_fires_with_the_result,
        test_no_hooks_restores_current_behavior_exactly,
        test_run_stop_fires_at_the_end_of_a_top_level_run,
        test_run_stop_routes_helper_ids_to_subagent_stop,
        test_runtime_shares_one_hook_manager_with_its_loop,
        test_prompt_submitted_fires_on_send_message,
        test_prompt_submitted_cannot_veto_the_run,
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
        print("FAIL - hooks:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - hooks: {len(tests)} tests "
          "(manager x7, pre/post/no-op tool points x3, run/subagent stop x2, "
          "prompt-submitted + wiring x3).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

"""Plan mode: a read-only investigation gate the user approves before the agent acts.

Step 19's promise is that a run put in plan mode can look but not touch - it may
read, search and reason, but writing a file, running a command, or leaving a note
that outlives the session is refused until the user has seen the plan and approved
it. "Approval" isn't a stored flag: it's simply the next run made *without* plan
mode, where full tools come back. The feature is composed from machinery that
already existed, so these tests check the two halves that make it a guarantee
rather than a hope:

  * the SOFT half - `AgentLoop._tool_schemas` narrows the tools the model is even
    shown to the read-only ones, and the request carries a banner telling it to
    plan rather than act (the banner rides on the wire only, never stored);
  * the HARD half - `PlanModePolicy` refuses any mutating call the model makes
    anyway (a name it hallucinated, a tool it saw in an earlier non-plan turn),
    and denial wins in the policy chain.

The end-to-end pair is the plan's own verify: the *same* scripted write call is
refused and hidden under plan mode - the file is never written, no prompt is shown
(it's a hard deny, not an ask) - and then runs and is visible when the next run
drops plan mode. That is "the agent cannot write or run a mutating command until
the plan is approved; approval flips it to full tools".

Offline by construction: a turn-aware stand-in model, no network, no key. Run under
pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src python3 packages/agent-native/tests/test_plan_mode.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any

from agent_native.config import AgentConfig
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    Session,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.loop import (
    _PLAN_MODE_BANNER,
    AgentLoop,
    Cancellation,
    Limits,
    RunContext,
    RunStatus,
    _with_plan_mode_banner,
)
from agent_native.models.base import ModelRegistry
from agent_native.permissions import (
    Decision,
    PermissionDecision,
    PlanModePolicy,
    Policy,
    PolicyChain,
)
from agent_native.tools.base import ToolRegistry
from agent_native.tools.manager import ToolManager

from tests._fake_tools import ReadFileTool, WriteFileTool
from tests._scripted import call_event, scripted_model, text_event


# ---------------------------------------------------------------------------
# Tiny doubles: a turn-aware model, an allow-all policy, a must-not-ask prompter
# ---------------------------------------------------------------------------
class TurnAwareProvider:
    """Turn one asks to write a file; every turn after finishes with text.

    `_scripted.ScriptedProvider` replays the *same* events each turn, which would
    make this run loop forever asking to write. Plan mode's whole point is a run
    that ends with a plan, so the model has to change what it says between turns.
    Like the scripted one it records `(messages, tools)` per call, so a test can
    read back exactly which tool schemas the model was offered and whether the
    plan-mode banner rode along on the wire.
    """

    def __init__(self, write_args: str) -> None:
        self._write_args = write_args
        self.requests: list = []
        self.closed = False
        self.calls = 0

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
    """A policy with no opinion - used to prove a later DENY still wins the chain."""

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        return Decision(PermissionDecision.ALLOW, reason="allow-all")


class _MustNotAsk:
    """A permission prompter that fails if used: plan mode denies, it never asks."""

    def __init__(self) -> None:
        self.asked = 0

    async def ask(self, request: Any, session_id: str) -> bool:
        self.asked += 1
        raise AssertionError("plan mode must refuse outright, never prompt the user")


# small definition/context stubs for the pure-unit policy tests
class _Perm:
    def __init__(self, read_only: bool) -> None:
        self.read_only = read_only


class _Defn:
    def __init__(self, read_only: bool) -> None:
        self.permissions = _Perm(read_only)


class _Ctx:
    def __init__(self, plan_mode: bool) -> None:
        self.limits = Limits(plan_mode=plan_mode)


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------
async def test_limits_carries_plan_mode_off_by_default() -> None:
    assert Limits().plan_mode is False              # ordinary runs are unaffected
    assert Limits(plan_mode=True).plan_mode is True  # opt in per run, like the ceilings


# ---------------------------------------------------------------------------
# The banner (soft half, the part the model is told)
# ---------------------------------------------------------------------------
async def test_banner_folds_into_the_system_message_without_mutating_the_wire() -> None:
    wire = [{"role": "system", "content": "BASE"}, {"role": "user", "content": "hi"}]
    out = _with_plan_mode_banner(wire)

    # Folded into the existing system message, not added as a new one, so the
    # message count and assistant/tool-result pairing are left as they were.
    assert len(out) == len(wire)
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("BASE")
    assert _PLAN_MODE_BANNER in out[0]["content"]
    # The user message is passed through untouched.
    assert out[1] is wire[1]
    # Ephemeral: the caller's wire (which `render` hands back fresh each turn) is
    # never mutated, so an approved later run replays a clean conversation.
    assert wire[0]["content"] == "BASE"


async def test_banner_is_inserted_when_there_is_no_system_message() -> None:
    out = _with_plan_mode_banner([{"role": "user", "content": "hi"}])
    assert out[0]["role"] == "system"
    assert _PLAN_MODE_BANNER in out[0]["content"]
    assert out[1]["role"] == "user"


# ---------------------------------------------------------------------------
# PlanModePolicy (hard half, the guarantee)
# ---------------------------------------------------------------------------
async def test_policy_allows_read_only_denies_mutating_while_planning() -> None:
    policy = PlanModePolicy()
    assert policy.check(_Ctx(True), _Defn(read_only=True), {}).result == PermissionDecision.ALLOW
    denied = policy.check(_Ctx(True), _Defn(read_only=False), {})
    assert denied.result == PermissionDecision.DENY
    assert "plan mode" in denied.reason.lower()


async def test_policy_is_inert_when_not_planning() -> None:
    policy = PlanModePolicy()
    # Off (the default): it has no opinion even about a mutating tool, so adding it
    # to the default chain changes nothing for an ordinary run.
    assert policy.check(_Ctx(False), _Defn(read_only=False), {}).result == PermissionDecision.ALLOW
    # Defensive: a missing context or missing limits reads as "not planning".
    assert policy.check(None, _Defn(read_only=False), {}).result == PermissionDecision.ALLOW


async def test_deny_wins_in_the_chain_regardless_of_order() -> None:
    mutating, planning = _Defn(read_only=False), _Ctx(True)
    assert (
        PolicyChain([_AllowAll(), PlanModePolicy()]).check(planning, mutating, {}).result
        == PermissionDecision.DENY
    )
    assert (
        PolicyChain([PlanModePolicy(), _AllowAll()]).check(planning, mutating, {}).result
        == PermissionDecision.DENY
    )
    # and when not planning, the allow-all passes straight through
    assert (
        PolicyChain([_AllowAll(), PlanModePolicy()]).check(_Ctx(False), mutating, {}).result
        == PermissionDecision.ALLOW
    )


# ---------------------------------------------------------------------------
# _tool_schemas (soft half, what the model is shown)
# ---------------------------------------------------------------------------
def _loop_with_fake_tools() -> tuple:
    """A loop wired to the read/write fake tools and a plan-mode-only policy chain.

    The same registry is handed to the loop (which reads it to decide *visible*
    tools) and to the ToolManager (which reads it to *find and run* them), so the
    soft and hard halves are talking about the same tools.
    """
    db = MemoryDatabase()
    provider = TurnAwareProvider('{"path": "out.txt", "content": "hello"}')
    registry = ModelRegistry()
    registry.register_provider("scripted", provider)
    registry.register_model("scripted-1", scripted_model())
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    prompter = _MustNotAsk()
    manager = ToolManager(tools, PolicyChain([PlanModePolicy()]), prompter)
    loop = AgentLoop(registry, tools, manager, ContextManager(), EventBus(db), db)
    return loop, db, provider, prompter


def _schema_names(schemas: list) -> set:
    return {s["function"]["name"] for s in schemas}


def _context(session: Session, plan_mode: bool) -> RunContext:
    return RunContext(
        session=session,
        run_id="run_plan",
        config=AgentConfig(model="scripted-1"),
        limits=Limits(max_turns=5, plan_mode=plan_mode),
        cancellation=Cancellation(),
    )


async def test_tool_schemas_hides_mutating_tools_in_plan_mode() -> None:
    loop, _db, _provider, _prompter = _loop_with_fake_tools()
    session = Session(agent="build", working_directory=".")

    full = _schema_names(loop._tool_schemas(_context(session, plan_mode=False)))
    planning = _schema_names(loop._tool_schemas(_context(session, plan_mode=True)))

    # Off: both tools are offered. On: only the read-only one survives.
    assert full == {"read_file", "write_file"}
    assert planning == {"read_file"}


# ---------------------------------------------------------------------------
# End to end: the plan's own verify, both sides of the gate
# ---------------------------------------------------------------------------
async def _run_write_attempt(plan_mode: bool) -> tuple:
    """Drive a full run whose model tries to write a file, and report what happened.

    Returns (result, wrote_path_exists, tools_offered_turn1, banner_seen_turn1,
    times_prompted) - everything the two e2e tests below assert on.
    """
    loop, db, provider, prompter = _loop_with_fake_tools()
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(agent="build", working_directory=tmp)
        await db.create_session(session)
        conv = Conversation([system_message("sys"), user_message(session.id, "make a file")])
        result = await loop.run(conv, _context(session, plan_mode=plan_mode))
        wrote = os.path.isfile(os.path.join(tmp, "out.txt"))
    messages_turn1, tools_turn1 = provider.requests[0]
    system_turn1 = messages_turn1[0].get("content", "") if messages_turn1 else ""
    return (
        result,
        wrote,
        _schema_names(tools_turn1),
        _PLAN_MODE_BANNER in system_turn1,
        prompter.asked,
    )


async def test_plan_mode_refuses_the_write_and_never_shows_the_tool() -> None:
    result, wrote, offered, banner_seen, prompted = await _run_write_attempt(plan_mode=True)

    # The run finished cleanly (it planned, then answered) rather than erroring.
    assert result.status is RunStatus.FINISHED
    # Hard half: the file was never written - the mutating tool did not run.
    assert wrote is False
    # ...and it was a hard deny, not a prompt the test happened to answer.
    assert prompted == 0
    # Soft half: the model was never even shown the writer, and was told to plan.
    assert offered == {"read_file"}
    assert banner_seen is True


async def test_dropping_plan_mode_lets_the_same_write_run() -> None:
    result, wrote, offered, banner_seen, prompted = await _run_write_attempt(plan_mode=False)

    # The very same scripted write call now goes through: approval = a run without
    # plan mode, where full tools are visible and the policy stays silent.
    assert result.status is RunStatus.FINISHED
    assert wrote is True
    assert prompted == 0                    # allowed outright, no prompt needed
    assert offered == {"read_file", "write_file"}
    assert banner_seen is False             # no banner on an ordinary run


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_limits_carries_plan_mode_off_by_default,
        test_banner_folds_into_the_system_message_without_mutating_the_wire,
        test_banner_is_inserted_when_there_is_no_system_message,
        test_policy_allows_read_only_denies_mutating_while_planning,
        test_policy_is_inert_when_not_planning,
        test_deny_wins_in_the_chain_regardless_of_order,
        test_tool_schemas_hides_mutating_tools_in_plan_mode,
        test_plan_mode_refuses_the_write_and_never_shows_the_tool,
        test_dropping_plan_mode_lets_the_same_write_run,
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
        print("FAIL - plan_mode:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - plan_mode: {len(tests)} tests "
          "(flag, banner x2, policy x3, tool visibility, e2e refuse, e2e allow).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

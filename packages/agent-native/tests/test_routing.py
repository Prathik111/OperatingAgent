"""Model routing + fallback: a strong model for the main loop, a cheaper one for
helpers, and an automatic hand-off to an alternate provider when the active one is
down.

Step 22's promise is *config-driven* routing - a list of fallback models on the
agent, tried in order - not a learned router. Two halves, and the plan's own verify
names both:

  * **Kill the primary provider mid-run and watch the fallback take over.** A run
    whose primary model keeps failing with a temporary error (a 429, an outage)
    fails over to the next model in its chain and finishes there, once the retry
    that already exists is exhausted (`test_fallback_takes_over_when_primary_fails`).
    The hand-off is sticky, so a dead provider isn't paid a full retry-and-timeout
    cycle at the top of every remaining turn (`test_fallover_is_sticky`).
  * **Subagents bill against the cheaper model in the receipt.** A helper given its
    own `model` runs on it, and its run record - the receipt - is priced at that
    model, not the parent's (`test_subagent_bills_against_its_cheaper_model`).

Everything else guards the machinery: the candidate list is the primary (resolved
strictly, so a bad *primary* is still a clean ERROR) plus the registered fallbacks
(a bad *fallback* skipped, never fatal), deduped; a permanent failure and a
cancelled run never fail over; and a run with no fallbacks is byte-for-byte the run
it always was, cost included.

Offline by construction: scripted stand-in models, no network, no key. Run under
pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src:packages/agent-native \
        python3 packages/agent-native/tests/test_routing.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from agent_native.config import AgentConfig, Subagent
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    Session,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus, EventType
from agent_native.loop import AgentLoop, Cancellation, Limits, RunContext, RunStatus
from agent_native.models.base import Model, ModelRegistry, StreamEvent, StreamType
from agent_native.permissions import Decision, PermissionDecision, Policy, PolicyChain
from agent_native.service import AgentRuntime
from agent_native.tools.base import (
    Tool,
    ToolDefinition,
    ToolPermissions,
    ToolRegistry,
    ToolResult,
)
from agent_native.tools.manager import ToolManager
from agent_native.tools.subagent import HELPER_RUN_SEPARATOR, DelegateTool

from tests._scripted import ScriptedProvider, call_event, text_event


# ---------------------------------------------------------------------------
# Stand-in providers whose failure mode the test owns
# ---------------------------------------------------------------------------
def _usage_event(input_tokens: int, output_tokens: int) -> StreamEvent:
    return StreamEvent(
        StreamType.USAGE, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


class _AlwaysRateLimited:
    """A provider that always fails, and fails *temporarily* - a 429, not a bad key.

    The message matters: it trips `_is_temporary`, so the loop retries it and then
    fails over. A permanent-looking error (a bad key) would correctly do neither,
    which is what `test_a_permanent_failure_does_not_fail_over` pins down.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        raise RuntimeError("rate limit reached: 429 too many requests")
        yield  # pragma: no cover - unreachable, but makes this an async generator

    def count_tokens(self, messages: list) -> int:
        return 0


class _AlwaysBadRequest:
    """A provider that always fails *permanently* - the request itself is wrong."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        raise RuntimeError("401 unauthorized: invalid api key")
        yield  # pragma: no cover - makes this an async generator

    def count_tokens(self, messages: list) -> int:
        return 0


class _AnswersWithUsage:
    """A provider that answers with text and reports usage, on every call."""

    def __init__(self, text: str = "backup answer", tokens: int = 1000) -> None:
        self.calls = 0
        self._text = text
        self._tokens = tokens

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        yield _usage_event(self._tokens, self._tokens)
        yield text_event(self._text)

    def count_tokens(self, messages: list) -> int:
        return 0


class _ToolThenText:
    """Turn one calls a named tool; every later turn answers with text.

    Enough to make a run take two turns, so a test can watch which model serves the
    *second* turn - the whole question a sticky fallover turns on.
    """

    def __init__(self, tool_name: str, text: str = "done") -> None:
        self.calls = 0
        self._tool = tool_name
        self._text = text

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        if self.calls == 1:
            yield call_event(0, self._tool, "{}")
        else:
            yield text_event(self._text)

    def count_tokens(self, messages: list) -> int:
        return 0


# ---------------------------------------------------------------------------
# Tiny doubles for the loop wiring
# ---------------------------------------------------------------------------
class _AllowAll(Policy):
    """Allows everything, so a run's only variable is the model, never the gate."""

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        return Decision(PermissionDecision.ALLOW, reason="allow-all")


class _MustNotAsk:
    """A prompter that fails if used: these runs must never stop to ask."""

    def __init__(self) -> None:
        self.asked = 0

    async def ask(self, request: Any, session_id: str) -> bool:
        self.asked += 1
        raise AssertionError("these runs allow everything and must never prompt")


class _NoopTool(Tool):
    """A read-only tool that does nothing - just something for a turn to call."""

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


def _model(name: str, provider: str, in_price: float = 0.0, out_price: float = 0.0) -> Model:
    return Model(
        provider=provider,
        model_id=name,
        context_size=100_000,
        input_price_per_million=in_price,
        output_price_per_million=out_price,
    )


def _loop(reg: ModelRegistry, tools: "ToolRegistry | None" = None):
    """A loop wired to `reg`, an allow-all gate, and a must-not-ask prompter."""
    db = MemoryDatabase()
    tools = tools if tools is not None else ToolRegistry()
    manager = ToolManager(tools, PolicyChain([_AllowAll()]), _MustNotAsk())
    loop = AgentLoop(reg, tools, manager, ContextManager(), EventBus(db), db)
    return loop, db


def _conversation(session: Session) -> Conversation:
    return Conversation(
        [system_message("You are a helpful assistant.", session.id),
         user_message(session.id, "do the thing")]
    )


def _context(session: Session, config: AgentConfig, max_turns: int = 3) -> RunContext:
    # max_retries=0 so a failing provider fails over at once instead of sleeping
    # through the retry backoff - the fallover is what's under test, not the wait.
    return RunContext(
        session=session,
        run_id="run_route",
        config=config,
        limits=Limits(max_turns=max_turns, max_retries=0),
        cancellation=Cancellation(),
    )


# ---------------------------------------------------------------------------
# The candidate list: primary strict, fallbacks best-effort, deduped
# ---------------------------------------------------------------------------
async def test_resolve_lists_primary_then_registered_fallbacks() -> None:
    reg = ModelRegistry()
    reg.register_provider("p", _AnswersWithUsage())
    reg.register_provider("b", _AnswersWithUsage())
    reg.register_model("primary-1", _model("primary-1", "p"))
    reg.register_model("backup-1", _model("backup-1", "b"))
    loop, _ = _loop(reg)

    # An unregistered fallback ("ghost-1") is skipped, never raised.
    config = AgentConfig(model="primary-1", fallback_models=["backup-1", "ghost-1"])
    candidates = loop._resolve_candidates(config)

    assert [m.model_id for m, _ in candidates] == ["primary-1", "backup-1"]


async def test_resolve_dedups_and_skips_a_fallback_with_no_provider() -> None:
    reg = ModelRegistry()
    reg.register_provider("p", _AnswersWithUsage())
    reg.register_provider("b", _AnswersWithUsage())
    reg.register_model("primary-1", _model("primary-1", "p"))
    reg.register_model("backup-1", _model("backup-1", "b"))
    # A model whose provider was never registered: resolvable by name, but its
    # provider lookup raises - so it's skipped, not fatal.
    reg.register_model("orphan-1", _model("orphan-1", "nowhere"))
    loop, _ = _loop(reg)

    config = AgentConfig(
        model="primary-1",
        # primary repeated, backup twice, and the orphan: dedup + skip leaves two.
        fallback_models=["primary-1", "backup-1", "backup-1", "orphan-1"],
    )
    candidates = loop._resolve_candidates(config)

    assert [m.model_id for m, _ in candidates] == ["primary-1", "backup-1"]


async def test_resolve_raises_on_a_bad_primary() -> None:
    """A bad *primary* is a config error the run must surface, not paper over."""
    reg = ModelRegistry()
    reg.register_provider("b", _AnswersWithUsage())
    reg.register_model("backup-1", _model("backup-1", "b"))
    loop, _ = _loop(reg)

    raised = False
    try:
        loop._resolve_candidates(AgentConfig(model="nope-1", fallback_models=["backup-1"]))
    except Exception:  # noqa: BLE001 - the exact type is the registry's, not ours
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# The plan's verify, half one: the fallback takes over mid-run
# ---------------------------------------------------------------------------
async def test_fallback_takes_over_when_primary_fails() -> None:
    """The primary 429s; the run fails over to the backup and finishes there."""
    primary = _AlwaysRateLimited()
    backup = _AnswersWithUsage(text="backup carried it", tokens=1000)
    reg = ModelRegistry()
    reg.register_provider("primary", primary)
    reg.register_provider("backup", backup)
    reg.register_model("primary-1", _model("primary-1", "primary", 1.0, 1.0))
    reg.register_model("backup-1", _model("backup-1", "backup", 5.0, 5.0))
    loop, db = _loop(reg)

    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    config = AgentConfig(model="primary-1", fallback_models=["backup-1"])
    result = await loop.run(_conversation(session), _context(session, config))

    # It finished, and it finished on the backup.
    assert result.status is RunStatus.FINISHED
    assert result.final_text == "backup carried it"
    assert result.model == "backup-1"
    assert result.fallbacks == 1
    assert result.retries == 0                 # max_retries=0: no re-sends, a straight fallover

    # Both were reached: the primary once (and failed), the backup once (and answered).
    assert primary.calls == 1
    assert backup.calls == 1

    # Priced against the model that actually served the turn - the backup - so the
    # receipt is honest about where the tokens were spent.
    assert result.cost_usd == round(_model("backup-1", "backup", 5.0, 5.0).cost_of(1000, 1000), 6)
    assert result.cost_usd == 0.01

    # And the hand-off is on the event stream, naming both ends, so a watcher can
    # see it happen rather than infer it from a model id that changed.
    events = await db.load_events(session.id, 0)
    handoffs = [e for e in events if e.type == EventType.MODEL_FALLBACK]
    assert len(handoffs) == 1
    assert handoffs[0].data["from_model"] == "primary-1"
    assert handoffs[0].data["to_model"] == "backup-1"


async def test_fallover_is_sticky() -> None:
    """Once failed over, later turns go straight to the backup - the primary is
    not paid a fresh retry-and-timeout cycle at the top of every turn."""
    primary = _AlwaysRateLimited()
    backup = _ToolThenText("noop", text="done after the tool")
    reg = ModelRegistry()
    reg.register_provider("primary", primary)
    reg.register_provider("backup", backup)
    reg.register_model("primary-1", _model("primary-1", "primary"))
    reg.register_model("backup-1", _model("backup-1", "backup"))

    tools = ToolRegistry()
    tools.register(_NoopTool())
    loop, db = _loop(reg, tools=tools)

    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    config = AgentConfig(model="primary-1", fallback_models=["backup-1"])
    result = await loop.run(_conversation(session), _context(session, config, max_turns=4))

    assert result.status is RunStatus.FINISHED
    assert result.model == "backup-1"
    assert result.fallbacks == 1              # failed over once, on turn one...
    assert result.turns == 2                  # ...and ran a second turn on the backup
    assert primary.calls == 1                 # the primary was tried once, never again
    assert backup.calls == 2                  # the backup served both turns


# ---------------------------------------------------------------------------
# Fallover only when it could help: not on a permanent error, not once streamed
# ---------------------------------------------------------------------------
async def test_a_permanent_failure_does_not_fail_over() -> None:
    """A bad key fails the same way on every model, so trying the backup is waste:
    the run errors on the primary without touching the fallback."""
    primary = _AlwaysBadRequest()
    backup = _AnswersWithUsage()
    reg = ModelRegistry()
    reg.register_provider("primary", primary)
    reg.register_provider("backup", backup)
    reg.register_model("primary-1", _model("primary-1", "primary"))
    reg.register_model("backup-1", _model("backup-1", "backup"))
    loop, db = _loop(reg)

    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    config = AgentConfig(model="primary-1", fallback_models=["backup-1"])
    result = await loop.run(_conversation(session), _context(session, config))

    assert result.status is RunStatus.ERROR
    assert result.fallbacks == 0
    assert primary.calls == 1
    assert backup.calls == 0                  # the backup was never reached


# ---------------------------------------------------------------------------
# A run with no fallbacks is the run it always was
# ---------------------------------------------------------------------------
async def test_a_run_without_fallbacks_is_unchanged() -> None:
    """No fallback chain: one model, priced exactly as before, zero fallovers.

    This guards the cost change - accumulating per turn against the serving model -
    against drifting away from the old 'price the whole total once' number for the
    ordinary single-model run. They're equal because pricing is linear.
    """
    provider = ScriptedProvider([_usage_event(1000, 1000), text_event("only answer")])
    reg = ModelRegistry()
    reg.register_provider("solo", provider)
    reg.register_model("solo-1", _model("solo-1", "solo", 2.0, 3.0))
    loop, db = _loop(reg)

    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    result = await loop.run(_conversation(session), _context(session, AgentConfig(model="solo-1")))

    assert result.status is RunStatus.FINISHED
    assert result.model == "solo-1"
    assert result.fallbacks == 0
    # Same number the old 'model.cost_of(total)' produced: (1000*2 + 1000*3)/1e6.
    assert result.cost_usd == round(_model("solo-1", "solo", 2.0, 3.0).cost_of(1000, 1000), 6)
    assert result.cost_usd == 0.005


# ---------------------------------------------------------------------------
# The plan's verify, half two: subagents bill against the cheaper model
# ---------------------------------------------------------------------------
async def test_subagent_bills_against_its_cheaper_model() -> None:
    """A helper with its own cheaper model runs on it, and its receipt is priced
    at that model - not the parent's strong one."""
    strong = _AnswersWithUsage(text="parent would answer")
    cheap = _AnswersWithUsage(text="helper done", tokens=1000)
    reg = ModelRegistry()
    reg.register_provider("strong", strong)
    reg.register_provider("cheap", cheap)
    reg.register_model("strong-1", _model("strong-1", "strong", 100.0, 100.0))
    reg.register_model("cheap-1", _model("cheap-1", "cheap", 1.0, 1.0))

    db = MemoryDatabase()
    parent_config = AgentConfig(
        name="build",
        model="strong-1",
        subagents=[Subagent(name="helper", model="cheap-1")],
    )
    runtime = AgentRuntime(database=db, model_registry=reg, agents=[parent_config])

    parent_session = Session(agent="build", working_directory=".")
    await db.create_session(parent_session)
    context = RunContext(
        session=parent_session,
        run_id="run_parent",
        config=runtime.config_for("build"),
        limits=Limits(max_turns=5),
        cancellation=Cancellation(),
    )

    result = await DelegateTool(runtime).execute({"helper": "helper", "job": "do it"}, context)
    assert result.success is True
    assert "helper done" in result.output

    # The helper's run receipt: its id is the parent's with a /helper suffix.
    helper_run = await db.get_run(f"run_parent{HELPER_RUN_SEPARATOR}helper")
    assert helper_run is not None
    assert helper_run.model == "cheap-1"                     # billed on its own model...

    cheap_cost = _model("cheap-1", "cheap", 1.0, 1.0).cost_of(1000, 1000)
    strong_cost = _model("strong-1", "strong", 100.0, 100.0).cost_of(1000, 1000)
    assert helper_run.cost_usd == round(cheap_cost, 6)       # ...at the cheap price...
    assert cheap_cost != strong_cost                         # ...which is not the strong price

    # The parent's own (strong) provider was never called - only the helper ran here.
    assert strong.calls == 0
    assert cheap.calls == 1


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_resolve_lists_primary_then_registered_fallbacks,
        test_resolve_dedups_and_skips_a_fallback_with_no_provider,
        test_resolve_raises_on_a_bad_primary,
        test_fallback_takes_over_when_primary_fails,
        test_fallover_is_sticky,
        test_a_permanent_failure_does_not_fail_over,
        test_a_run_without_fallbacks_is_unchanged,
        test_subagent_bills_against_its_cheaper_model,
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
        print("FAIL - routing:")
        for line in failures:
            print("  -", line)
        return 1
    print(
        f"PASS - routing: {len(tests)} tests (candidate list x3, fallover takes over, "
        "sticky, permanent-no-fallover, no-fallback-unchanged, subagent billing)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

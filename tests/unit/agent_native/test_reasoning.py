"""The thinking-budget knob, and the reasoning-token count it makes visible.

Step 21 adds one small control - `reasoning_effort` - in three places that have to
agree: the agent's default (`AgentConfig`), the per-run override (`Limits`), and
the value that actually reaches the provider. And it adds one number back on the
receipt - `reasoning_tokens` - which is a *breakdown* of the output tokens, not an
addition to them, so it must never move the total or the cost.

What's checked here, all offline against stand-in providers whose timing and token
counts the test owns:

  * raising the budget raises the reasoning-token count on the receipt, and the
    same number rides the RUN_FINISHED event a UI watches;
  * that count is a slice of `output_tokens`, so `total_tokens` is unmoved by it;
  * `Limits.reasoning_effort` wins over `AgentConfig.reasoning_effort`, and the
    agent default is used when the run is silent;
  * a provider with no thinking mode is handed a non-empty budget and finishes
    cleanly with the count at zero - "ignores it without error";
  * a provider written before the knob existed (the old four-arg `stream`) is
    still called the old way when no effort is set, so nothing that didn't ask
    for the feature is broken by it;
  * the CLI surfaces it: the receipt line, the history totals, the runs table.

Run under pytest, or straight (the __main__ block) on a box without pytest:
    PYTHONPATH=packages/agent-native/src python3 packages/agent-native/tests/test_reasoning.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, ClassVar

from agent_native.config import AgentConfig
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    Session,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus, EventType
from agent_native.loop import (
    AgentLoop,
    Cancellation,
    Limits,
    RunContext,
    RunRecord,
    RunStatus,
)
from agent_native.main import _receipt, _render_runs_table, _runs_totals
from agent_native.models.base import Model, ModelRegistry, StreamEvent, StreamType
from agent_native.tools.base import ToolRegistry

from tests._scripted import ScriptedProvider, text_event


class ReasoningProvider:
    """A stand-in whose thinking scales with the budget it's handed.

    It records every effort it was called with, so a test can prove the right
    value reached it, and reports `reasoning_tokens` that grow with the budget -
    always as a slice of the `output_tokens` it also reports, the way a real
    reasoning model bills it. No network, no key.
    """

    _SCALE: ClassVar[dict[str, int]] = {
        "": 0,
        "low": 10,
        "medium": 50,
        "high": 200,
    }

    def __init__(self) -> None:
        self.efforts: list = []
        self.closed = False

    async def stream(
        self,
        messages: list,
        tools: list,
        model: Any,
        temperature: float = 0.0,
        reasoning_effort: str = "",
    ):
        self.efforts.append(reasoning_effort)
        reasoning = self._SCALE.get(reasoning_effort, 0)
        try:
            if reasoning:
                yield StreamEvent(StreamType.REASONING, {"text": "think. " * (reasoning // 10)})
            yield StreamEvent(StreamType.TEXT, {"text": "the answer"})
            yield StreamEvent(
                StreamType.USAGE,
                {
                    "input_tokens": 30,
                    # reasoning is part of the output count, never on top of it
                    "output_tokens": reasoning + 20,
                    "cached_tokens": 0,
                    "reasoning_tokens": reasoning,
                },
            )
            yield StreamEvent(StreamType.DONE, {"finish_reason": "stop"})
        finally:
            self.closed = True

    def count_tokens(self, messages: list) -> int:
        return 0


class PlainProvider:
    """A model with no thinking mode: accepts the knob, reports no reasoning.

    Stands for the provider-without-the-feature case. It must take a non-empty
    effort without complaint and leave `reasoning_tokens` unset, so the count
    stays honestly at zero rather than being invented.
    """

    def __init__(self) -> None:
        self.efforts: list = []
        self.closed = False

    async def stream(
        self,
        messages: list,
        tools: list,
        model: Any,
        temperature: float = 0.0,
        reasoning_effort: str = "",
    ):
        self.efforts.append(reasoning_effort)
        try:
            yield StreamEvent(StreamType.TEXT, {"text": "done"})
            # No reasoning_tokens key at all - the loop must keep it at zero.
            yield StreamEvent(
                StreamType.USAGE,
                {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0},
            )
            yield StreamEvent(StreamType.DONE, {"finish_reason": "stop"})
        finally:
            self.closed = True

    def count_tokens(self, messages: list) -> int:
        return 0


def _registry(provider: Any) -> ModelRegistry:
    """A registry wired to one provider under the model name the tests use."""
    registry = ModelRegistry()
    registry.register_provider("scripted", provider)
    registry.register_model(
        "scripted-1", Model(provider="scripted", model_id="scripted-1", context_size=100_000)
    )
    return registry


def _loop(provider: Any) -> tuple:
    db = MemoryDatabase()
    return (
        AgentLoop(_registry(provider), ToolRegistry(), None, ContextManager(), EventBus(db), db),
        db,
    )


async def _run(provider: Any, config: AgentConfig, limits: Limits) -> tuple:
    """Run one simple no-tool turn and hand back (result, db, session)."""
    loop, db = _loop(provider)
    session = Session(agent="build")
    await db.create_session(session)
    conv = Conversation([system_message("sys"), user_message(session.id, "hi")])
    context = RunContext(
        session=session,
        run_id="run_test",
        config=config,
        limits=limits,
        cancellation=Cancellation(),
    )
    result = await loop.run(conv, context)
    return result, db, session


# ---------------------------------------------------------------------------
# The budget changes the number on the receipt
# ---------------------------------------------------------------------------
async def test_more_effort_yields_more_reasoning_tokens() -> None:
    low_result, _, _ = await _run(
        ReasoningProvider(), AgentConfig(model="scripted-1"), Limits(reasoning_effort="low")
    )
    high_result, _, _ = await _run(
        ReasoningProvider(), AgentConfig(model="scripted-1"), Limits(reasoning_effort="high")
    )

    assert low_result.status is RunStatus.FINISHED
    assert high_result.status is RunStatus.FINISHED
    # Raising the budget raised the reasoning count on the receipt.
    assert high_result.usage.reasoning_tokens > low_result.usage.reasoning_tokens
    assert low_result.usage.reasoning_tokens == 10
    assert high_result.usage.reasoning_tokens == 200
    # And it stayed a slice of the output, never larger than it.
    assert high_result.usage.reasoning_tokens <= high_result.usage.output_tokens


async def test_reasoning_tokens_are_a_breakdown_not_an_addition() -> None:
    """The whole point of the field: cost and total are unmoved by it."""
    result, _, _ = await _run(
        ReasoningProvider(), AgentConfig(model="scripted-1"), Limits(reasoning_effort="high")
    )
    usage = result.usage
    assert usage.reasoning_tokens == 200
    # total_tokens is input+output only - the reasoning slice is already inside output.
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.total_tokens == 30 + 220


async def test_reasoning_tokens_reach_the_run_finished_event() -> None:
    """A UI that only watches the event stream must see the same number."""
    result, db, session = await _run(
        ReasoningProvider(), AgentConfig(model="scripted-1"), Limits(reasoning_effort="medium")
    )
    events = await db.load_events(session.id)
    finished = [e for e in events if e.type == EventType.RUN_FINISHED]
    assert len(finished) == 1
    data = finished[0].data
    assert data["reasoning_tokens"] == result.usage.reasoning_tokens == 50
    # The receipt on the event carries the same output count the run did.
    assert data["output_tokens"] == result.usage.output_tokens


# ---------------------------------------------------------------------------
# Where the effort comes from: config default vs per-run override
# ---------------------------------------------------------------------------
async def test_agent_config_effort_is_used_when_the_run_is_silent() -> None:
    provider = ReasoningProvider()
    result, _, _ = await _run(
        provider,
        AgentConfig(model="scripted-1", reasoning_effort="high"),
        Limits(max_turns=4),  # no reasoning_effort set on the run
    )
    assert result.status is RunStatus.FINISHED
    assert provider.efforts == ["high"]  # the agent's default reached the provider
    assert result.usage.reasoning_tokens == 200


async def test_limits_effort_overrides_the_agent_config() -> None:
    provider = ReasoningProvider()
    result, _, _ = await _run(
        provider,
        AgentConfig(model="scripted-1", reasoning_effort="low"),
        Limits(reasoning_effort="high"),  # the run asks for more than the agent's default
    )
    assert result.status is RunStatus.FINISHED
    assert provider.efforts == ["high"]  # the per-run value won
    assert result.usage.reasoning_tokens == 200


async def test_no_effort_anywhere_passes_nothing_to_a_legacy_provider() -> None:
    """The old four-arg `stream` (no reasoning_effort param) must still be callable.

    When neither the agent nor the run sets an effort, the loop passes no keyword
    at all - so a provider written before the knob existed is called exactly the
    way it always was. `ScriptedProvider` is such a provider; if the loop handed
    it a `reasoning_effort=` it didn't declare, this would raise TypeError.
    """
    provider = ScriptedProvider([text_event("done")])
    result, _, _ = await _run(provider, AgentConfig(model="scripted-1"), Limits(max_turns=1))
    assert result.status is RunStatus.FINISHED
    assert "done" in result.final_text


# ---------------------------------------------------------------------------
# A provider without a thinking mode
# ---------------------------------------------------------------------------
async def test_a_provider_without_thinking_ignores_the_budget() -> None:
    provider = PlainProvider()
    result, _, _ = await _run(
        provider, AgentConfig(model="scripted-1"), Limits(reasoning_effort="high")
    )
    # It accepted the budget (saw "high") but finished normally...
    assert provider.efforts == ["high"]
    assert result.status is RunStatus.FINISHED
    # ...and reported no reasoning, honestly, rather than inventing a number.
    assert result.usage.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# The CLI surfaces it: receipt line, history totals, runs table
# ---------------------------------------------------------------------------
def test_receipt_shows_reasoning_only_when_present() -> None:
    with_reasoning = _receipt(
        {"status": "finished", "turns": 1, "input_tokens": 30, "output_tokens": 220,
         "reasoning_tokens": 200, "cost_usd": 0.0}
    )
    assert "(200 reasoning)" in with_reasoning

    without = _receipt(
        {"status": "finished", "turns": 1, "input_tokens": 30, "output_tokens": 20,
         "reasoning_tokens": 0, "cost_usd": 0.0}
    )
    assert "reasoning" not in without

    # A pre-Step-21 receipt dict with no reasoning key at all must not crash.
    legacy = _receipt(
        {"status": "finished", "turns": 1, "input_tokens": 30, "output_tokens": 20, "cost_usd": 0.0}
    )
    assert "reasoning" not in legacy


def test_runs_totals_sums_reasoning() -> None:
    runs = [
        RunRecord(run_id="r1", session_id="s", status="finished", reasoning_tokens=10),
        RunRecord(run_id="r2", session_id="s", status="finished", reasoning_tokens=200),
        RunRecord(run_id="r3", session_id="s", status="finished", reasoning_tokens=0),
    ]
    totals = _runs_totals(runs)
    assert totals["reasoning_tokens"] == 210


def test_render_table_shows_reasoning_column_only_when_present() -> None:
    with_reasoning = [
        RunRecord(run_id="r1", session_id="s", status="finished", output_tokens=220,
                  reasoning_tokens=200),
    ]
    table = _render_runs_table(with_reasoning)
    header = table.splitlines()[0]
    assert "REASONING" in header
    totals_line = next(line for line in table.splitlines() if line.startswith("TOTALS"))
    assert "200" in totals_line

    without = [RunRecord(run_id="r2", session_id="s", status="finished", output_tokens=20)]
    assert "REASONING" not in _render_runs_table(without).splitlines()[0]


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    sync_tests = [
        test_receipt_shows_reasoning_only_when_present,
        test_runs_totals_sums_reasoning,
        test_render_table_shows_reasoning_column_only_when_present,
    ]
    async_tests = [
        test_more_effort_yields_more_reasoning_tokens,
        test_reasoning_tokens_are_a_breakdown_not_an_addition,
        test_reasoning_tokens_reach_the_run_finished_event,
        test_agent_config_effort_is_used_when_the_run_is_silent,
        test_limits_effort_overrides_the_agent_config,
        test_no_effort_anywhere_passes_nothing_to_a_legacy_provider,
        test_a_provider_without_thinking_ignores_the_budget,
    ]
    failures: list = []
    for test in sync_tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    for test in async_tests:
        try:
            asyncio.run(test())
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - reasoning:")
        for line in failures:
            print("  -", line)
        return 1
    total = len(sync_tests) + len(async_tests)
    print(f"PASS - reasoning: {total} tests "
          "(budget->tokens, breakdown-not-addition, event, precedence, "
          "no-op provider, legacy signature, CLI surfaces).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

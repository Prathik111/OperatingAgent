"""Budget governance: a run stops at its ceiling, cleanly, and says which one.

Step 9's promise is that a run given a small budget stops *at* it - a partial
result and a named reason, never a runaway bill and never an error. The ceilings
are checked between turns on the usage the provider reported, so these tests script
a turn that always asks for one more tool call (so the run would go forever) and
prove a ceiling is what stops it:

  * a token ceiling stops the run and names "max_tokens";
  * a cost ceiling stops it and names "max_cost";
  * either way the receipt still carries the turn it managed and the transcript
    still holds the partial work - "it still returns what it had", the plan's verify;
  * with every ceiling left at 0 (the default) nothing interferes: a normal finish
    names no reason.

The last test covers the step's other half - coordinating backoff so a shared rate
limit isn't retried N-ways at once - by checking `RetryCoordinator` serialises
concurrent backoffs.

Offline by construction: the model is `_scripted.ScriptedProvider`, no network, no
key. Run under pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src python3 tests/unit/agent_native/test_budget.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from agent_native.config import AgentConfig
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    Role,
    Session,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.loop import (
    AgentLoop,
    Cancellation,
    Limits,
    RetryCoordinator,
    RunContext,
    RunStatus,
)
from agent_native.models.base import Model, ModelRegistry, StreamEvent, StreamType
from agent_native.tools.base import ToolRegistry, ToolResult

from tests._scripted import ScriptedProvider, call_event, text_event


class NoopToolManager:
    """Authorizes everything, runs nothing that matters - a turn just needs a tool
    to run so the loop keeps going until a ceiling stops it."""

    async def authorize(self, call: Any, context: Any) -> Any:
        return None  # None means "not refused"

    async def run_authorized(self, call: Any, context: Any) -> ToolResult:
        return ToolResult(True, output="ok")


def _model(input_price: float = 0.0, output_price: float = 0.0) -> Model:
    """A scripted model, optionally priced so `cost_of` returns real dollars.

    Left unpriced (the default) the cost stays zero, so only a token ceiling can
    trip - which is exactly what the token test wants.
    """
    return Model(
        provider="scripted",
        model_id="scripted-1",
        context_size=100_000,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )


def _loop(provider: ScriptedProvider, model: Model, tool_manager: Any) -> tuple:
    db = MemoryDatabase()
    registry = ModelRegistry()
    registry.register_provider("scripted", provider)
    registry.register_model("scripted-1", model)
    loop = AgentLoop(
        registry, ToolRegistry(), tool_manager, ContextManager(), EventBus(db), db
    )
    return loop, db


def _context(session: Session, limits: Limits) -> RunContext:
    return RunContext(
        session=session,
        run_id="run_budget",
        config=AgentConfig(model="scripted-1"),
        limits=limits,
        cancellation=Cancellation(),
    )


def _usage_event(input_tokens: int, output_tokens: int) -> StreamEvent:
    return StreamEvent(
        StreamType.USAGE, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


def _working_turn(in_tokens: int = 1000, out_tokens: int = 1000) -> list:
    """A turn that does a little work and reports its usage, and always asks for one
    more tool call - so the run never ends on its own and a ceiling is what stops it."""
    return [text_event("thinking"), call_event(0, "noop", "{}"), _usage_event(in_tokens, out_tokens)]


async def _seeded(model: Model) -> tuple:
    provider = ScriptedProvider(_working_turn(1000, 1000))
    loop, db = _loop(provider, model, NoopToolManager())
    session = Session(agent="build")
    await db.create_session(session)
    conv = Conversation([system_message("sys"), user_message(session.id, "go")])
    return loop, db, session, conv


# ---------------------------------------------------------------------------
# The ceilings
# ---------------------------------------------------------------------------
async def test_token_ceiling_stops_cleanly_and_names_the_reason() -> None:
    # Unpriced model, so cost stays zero and only the token ceiling can trip.
    loop, _db, session, conv = await _seeded(_model())
    # A turn is 2000 tokens; a 1500 ceiling trips on the check before turn two.
    limits = Limits(max_turns=50, max_total_tokens=1500)

    result = await loop.run(conv, _context(session, limits))

    assert result.status is RunStatus.LIMIT_REACHED
    assert result.stop_reason == "max_tokens"
    assert result.turns == 1                       # stopped before the second call
    assert result.usage.input_tokens == 1000       # and kept the turn it did make
    assert result.usage.output_tokens == 1000


async def test_cost_ceiling_stops_cleanly_and_names_the_reason() -> None:
    # $1 per million each way, so one 2000-token turn costs $0.002.
    loop, _db, session, conv = await _seeded(_model(input_price=1.0, output_price=1.0))
    limits = Limits(max_turns=50, max_cost_usd=0.001)  # already over after one turn

    result = await loop.run(conv, _context(session, limits))

    assert result.status is RunStatus.LIMIT_REACHED
    assert result.stop_reason == "max_cost"
    assert result.turns == 1
    assert result.cost_usd >= 0.001                # the receipt shows what it spent


async def test_a_stopped_run_returns_the_partial_work() -> None:
    """The plan's verify: a capped run 'still returns what it had'."""
    loop, db, session, conv = await _seeded(_model())
    result = await loop.run(conv, _context(session, Limits(max_turns=50, max_total_tokens=1500)))

    # The receipt carries the turn it managed before the ceiling.
    assert result.turns == 1
    assert result.usage.input_tokens + result.usage.output_tokens == 2000
    # The partial work is in the transcript, not thrown away.
    assistant = [m for m in conv.messages if m.role == Role.ASSISTANT]
    assert assistant and "thinking" in assistant[-1].text()
    # And a receipt was still saved, so the history view sees the capped run.
    saved = await db.get_run(result.run_id)
    assert saved is not None and saved.status == "limit_reached"


async def test_no_ceilings_by_default_finish_names_no_reason() -> None:
    # A text-only turn (no tool call) finishes on its own; every ceiling is 0 = off.
    provider = ScriptedProvider([text_event("all done")])
    loop, db = _loop(provider, _model(), NoopToolManager())
    session = Session(agent="build")
    await db.create_session(session)
    conv = Conversation([system_message("sys"), user_message(session.id, "go")])

    result = await loop.run(conv, _context(session, Limits()))

    assert result.status is RunStatus.FINISHED
    assert result.stop_reason == ""                # a normal finish names no ceiling
    assert result.final_text == "all done"


# ---------------------------------------------------------------------------
# Coordinating backoff
# ---------------------------------------------------------------------------
async def test_retry_coordinator_serialises_concurrent_backoffs() -> None:
    """Two backoffs at once don't overlap: the second waits for the first.

    That's what keeps a shared rate limit from being retried N-ways in the same
    instant - the sleeps are staggered, not simultaneous. Serialised, four backoffs
    of `delay` take about 4*delay; run concurrently they'd take about one.
    """
    coordinator = RetryCoordinator()
    delay = 0.05
    start = time.monotonic()
    await asyncio.gather(*(coordinator.backoff(delay) for _ in range(4)))
    elapsed = time.monotonic() - start
    assert elapsed >= 3 * delay


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_token_ceiling_stops_cleanly_and_names_the_reason,
        test_cost_ceiling_stops_cleanly_and_names_the_reason,
        test_a_stopped_run_returns_the_partial_work,
        test_no_ceilings_by_default_finish_names_no_reason,
        test_retry_coordinator_serialises_concurrent_backoffs,
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
        print("FAIL - budget:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - budget: {len(tests)} tests "
          "(token ceiling, cost ceiling, partial result, ceilings-off, backoff coordination).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

"""Tests for the verifier node (``agent_langgraph.nodes.verifier``).

The verifier is the commit point: on a passing verdict it marks the step
verified and advances ``current_step``; on a failing verdict it leaves the
pointer put so the same index re-runs. It has three layers — deterministic
checks, an optional LLM judgement, and a fail-closed fallback — all exercised
here.
"""

from __future__ import annotations

from agent_langgraph.nodes.verifier import VerifierNode
from common.enums import RunStatus, TaskStatus, VerificationResult

from tests.support.langgraph import (
    StubModel,
    build_agent_config,
    build_context,
    build_runtime,
    make_plan,
    make_state,
    make_step,
)


def run_verifier(config, state, *, model=None, prompt_manager=None):
    context = build_context(config, model=model, prompt_manager=prompt_manager)
    return VerifierNode(state, build_runtime(context))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


async def test_no_plan_routes_to_responding(agent_config) -> None:
    delta = await run_verifier(agent_config, make_state(plan=None))
    assert delta["status"] is TaskStatus.RESPONDING


async def test_pointer_past_end_routes_to_responding(agent_config) -> None:
    state = make_state(plan=make_plan(make_step(1)), current_step=5)
    delta = await run_verifier(agent_config, state)
    assert delta["status"] is TaskStatus.RESPONDING


# ---------------------------------------------------------------------------
# Deterministic layer (no LLM round-trip)
# ---------------------------------------------------------------------------


async def test_failed_step_rejected_without_llm(agent_config) -> None:
    """A step that failed in execution is rejected deterministically, and
    retry_count is NOT incremented (the executor already counted it)."""
    step = make_step(1, status=RunStatus.FAILED, output="tool error")
    model = StubModel(verdict_success=True)  # would pass if consulted
    delta = await run_verifier(
        agent_config, make_state(plan=make_plan(step), current_step=0, retry_count=1),
        model=model,
    )
    assert delta["verification_success"] is False
    assert delta["retry_count"] == 1  # unchanged
    assert model.structured_calls == []  # LLM never consulted


async def test_empty_output_rejected_and_counted(agent_config) -> None:
    """An empty output is rejected deterministically; since the step is not
    FAILED, this rejection increments retry_count."""
    step = make_step(1, status=RunStatus.COMPLETED, output="   ")
    delta = await run_verifier(
        agent_config, make_state(plan=make_plan(step), current_step=0, retry_count=0)
    )
    assert delta["verification_success"] is False
    assert delta["retry_count"] == 1


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------


def completed_step_state(**overrides):
    step = make_step(1, status=RunStatus.COMPLETED, output="a real result")
    return make_state(plan=make_plan(step), current_step=0, **overrides)


async def test_llm_pass_advances_pointer_and_marks_verified(agent_config) -> None:
    model = StubModel(verdict_success=True, verdict_reason="looks right")
    plan = make_plan(make_step(1, status=RunStatus.COMPLETED, output="ok"),
                     make_step(2, status=RunStatus.COMPLETED, output="ok"))
    delta = await run_verifier(agent_config, make_state(plan=plan, current_step=0), model=model)

    assert delta["verification_success"] is True
    assert delta["current_step"] == 1  # advanced
    assert delta["status"] is TaskStatus.EXECUTING  # more steps remain
    verified = delta["plan"].steps[0]
    assert verified.verified is True
    assert verified.verification is VerificationResult.VERIFIED
    assert delta["verification_reason"] == "looks right"


async def test_llm_pass_on_last_step_routes_to_responding(agent_config) -> None:
    model = StubModel(verdict_success=True)
    delta = await run_verifier(agent_config, completed_step_state(), model=model)
    assert delta["current_step"] == 1
    assert delta["status"] is TaskStatus.RESPONDING


async def test_llm_reject_holds_pointer_and_counts(agent_config) -> None:
    model = StubModel(verdict_success=False, verdict_reason="off target")
    delta = await run_verifier(agent_config, completed_step_state(retry_count=0), model=model)

    assert delta["verification_success"] is False
    assert "current_step" not in delta  # pointer held
    assert delta["retry_count"] == 1
    assert delta["plan"].steps[0].verification is VerificationResult.NOT_VERIFIED
    assert delta["last_error"] == "off target"


async def test_llm_failure_fails_closed(agent_config) -> None:
    """If the verifier model raises, the step is rejected — never silently
    passed as verified."""
    model = StubModel(structured_error=RuntimeError("model down"))
    delta = await run_verifier(agent_config, completed_step_state(retry_count=0), model=model)
    assert delta["verification_success"] is False
    assert "verifier unavailable" in delta["verification_reason"]


# ---------------------------------------------------------------------------
# Verification disabled
# ---------------------------------------------------------------------------


async def test_verification_disabled_passes_without_llm() -> None:
    config = build_agent_config(require_verification=False)
    model = StubModel(verdict_success=False)  # would fail if consulted
    delta = await run_verifier(config, completed_step_state(), model=model)

    assert delta["verification_success"] is True
    assert delta["verification_reason"] == "verification disabled"
    assert model.structured_calls == []  # LLM skipped entirely

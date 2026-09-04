"""Unit tests for the responder node (``agent_langgraph.nodes.responder``).

The responder is terminal on every path — success, trivial plan, or give-up —
and must never fabricate success; it falls back to a deterministic summary if
the model or prompt is unavailable.
"""

from __future__ import annotations

import pytest
from agent_langgraph.nodes.responder import ResponderNode
from common.enums import RunStatus, TaskStatus, VerificationResult

from tests.support.langgraph import (
    StubModel,
    StubPromptManager,
    build_context,
    build_runtime,
    make_plan,
    make_state,
    make_step,
)


def run_responder(config, state, *, model=None, prompt_manager=None):
    context = build_context(config, model=model, prompt_manager=prompt_manager)
    return ResponderNode(state, build_runtime(context))


def verified_step(step_id: int = 1, output: str = "result"):
    return make_step(
        step_id, status=RunStatus.COMPLETED, verified=True,
        verification=VerificationResult.VERIFIED, output=output,
    )


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


async def test_responder_success_marks_completed(agent_config) -> None:
    answer = "All done."
    state = make_state(plan=make_plan(verified_step()), last_error=None)
    delta = await run_responder(agent_config, state, model=StubModel(answer=answer))

    assert delta["status"] is TaskStatus.COMPLETED
    assert delta["messages"][0].content == answer


async def test_responder_trivial_empty_plan_succeeds(agent_config) -> None:
    model = StubModel(answer="Nothing needed doing.")
    state = make_state(plan=make_plan(), last_error=None)
    delta = await run_responder(agent_config, state, model=model)
    assert delta["status"] is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Failure paths — the responder must never dress a failure up as success
# ---------------------------------------------------------------------------


async def test_responder_failure_marks_failed(agent_config) -> None:
    model = StubModel(answer="Sorry, it did not work.")
    state = make_state(plan=make_plan(make_step(1, status=RunStatus.FAILED)),
                       last_error="boom")
    delta = await run_responder(agent_config, state, model=model)
    assert delta["status"] is TaskStatus.FAILED


@pytest.mark.regression
async def test_responder_last_error_forces_failure_even_if_steps_ok(agent_config) -> None:
    """A lingering last_error means the run did not cleanly succeed, regardless
    of step state."""
    model = StubModel(answer="mixed")
    state = make_state(plan=make_plan(verified_step()), last_error="late failure")
    delta = await run_responder(agent_config, state, model=model)
    assert delta["status"] is TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Fallback synthesis
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_responder_uses_fallback_when_prompt_unavailable(agent_config) -> None:
    """If the prompt/model path raises, a deterministic summary is produced so
    the graph always terminates with something meaningful."""
    prompt_manager = StubPromptManager(error=FileNotFoundError("no responder.txt"))
    state = make_state(plan=make_plan(verified_step(output="the answer")), last_error=None)
    delta = await run_responder(agent_config, state, prompt_manager=prompt_manager)

    assert delta["status"] is TaskStatus.COMPLETED
    assert "the answer" in delta["messages"][0].content


@pytest.mark.regression
async def test_responder_fallback_reports_failure_honestly(agent_config) -> None:
    prompt_manager = StubPromptManager(error=FileNotFoundError("missing"))
    state = make_state(plan=make_plan(make_step(1, status=RunStatus.FAILED)),
                       last_error="disk full")
    delta = await run_responder(agent_config, state, prompt_manager=prompt_manager)
    assert delta["status"] is TaskStatus.FAILED
    assert "disk full" in delta["messages"][0].content


async def test_responder_empty_model_answer_falls_back(agent_config) -> None:
    model = StubModel(answer="   ")  # whitespace-only -> not acceptable
    state = make_state(plan=make_plan(verified_step(output="data")), last_error=None)
    delta = await run_responder(agent_config, state, model=model)
    assert "data" in delta["messages"][0].content

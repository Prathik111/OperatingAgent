"""Tests for the graph routing functions (``agent_langgraph.graph.routing``).

These functions are pure — state in, next-node out — and encode the control
flow of the whole graph, so each branch is pinned explicitly. The key
invariants: the verifier owns the step-pointer advance, and any recoverable
failure — a tool error or a rejected result — is handed back to the planner to
modify the approach, since a verbatim re-run would not help.
"""

from __future__ import annotations

from agent_langgraph.graph.constants import (
    ERROR_HANDLER,
    EXECUTOR,
    MAX_RETRIES,
    PHASE_TRANSITION,
    PLANNER,
    RESPONDER,
)
from agent_langgraph.graph.routing import (
    phase_router,
    retry_router,
    should_execute,
    verification_router,
)
from common.enums import RunStatus, WorkflowPhase

from tests.support.langgraph import make_plan, make_state, make_step

# ---------------------------------------------------------------------------
# should_execute (after the planner)
# ---------------------------------------------------------------------------


def test_should_execute_routes_to_executor_when_steps_remain() -> None:
    state = make_state(plan=make_plan(make_step(1), make_step(2)), current_step=0)
    assert should_execute(state) == EXECUTOR


def test_should_execute_routes_to_responder_when_plan_empty() -> None:
    """An empty plan has no execution work and can be answered immediately."""
    state = make_state(plan=make_plan(), current_step=0)
    assert should_execute(state) == RESPONDER


def test_should_execute_routes_to_responder_when_pointer_past_end() -> None:
    state = make_state(plan=make_plan(make_step(1)), current_step=1)
    assert should_execute(state) == RESPONDER


def test_should_execute_tolerates_missing_plan() -> None:
    state = make_state(plan=None, current_step=0)
    assert should_execute(state) == RESPONDER


# ---------------------------------------------------------------------------
# verification_router (after the verifier)
# ---------------------------------------------------------------------------


def test_verification_failure_routes_to_error_handler() -> None:
    state = make_state(verification_success=False)
    assert verification_router(state) == ERROR_HANDLER


def test_verification_missing_key_treated_as_failure() -> None:
    state = make_state()
    state.pop("verification_success", None)
    assert verification_router(state) == ERROR_HANDLER


def test_verification_success_with_more_steps_routes_to_executor() -> None:
    state = make_state(
        plan=make_plan(make_step(1), make_step(2)),
        current_step=1,  # verifier already advanced the pointer
        verification_success=True,
    )
    assert verification_router(state) == EXECUTOR


def test_verification_success_with_plan_exhausted_routes_to_phase_transition() -> None:
    """The verifier no longer ends the run directly: an exhausted plan hands off
    to the phase transition, which may start a remediation phase."""
    state = make_state(
        plan=make_plan(make_step(1)),
        current_step=1,
        verification_success=True,
    )
    assert verification_router(state) == PHASE_TRANSITION


# ---------------------------------------------------------------------------
# phase_router (after the phase transition)
# ---------------------------------------------------------------------------


def test_phase_router_sends_remediate_to_planner() -> None:
    state = make_state(workflow_phase=WorkflowPhase.REMEDIATE)
    assert phase_router(state) == PLANNER


def test_phase_router_sends_complete_to_responder() -> None:
    state = make_state(workflow_phase=WorkflowPhase.COMPLETE)
    assert phase_router(state) == RESPONDER


def test_phase_router_treats_missing_phase_as_more_work() -> None:
    """Defensive: an unset phase means the run hasn't finished planning yet."""
    state = make_state()
    state.pop("workflow_phase", None)
    assert phase_router(state) == PLANNER


# ---------------------------------------------------------------------------
# retry_router (after the error handler)
# ---------------------------------------------------------------------------


def test_retry_router_gives_up_when_budget_spent() -> None:
    state = make_state(retry_count=MAX_RETRIES)
    assert retry_router(state) == RESPONDER


def test_retry_router_replans_on_execution_failure() -> None:
    """A tool that threw leaves the step FAILED -> replan around it. A verbatim
    re-run would just fail again, and transient faults were already retried
    inside the executor, so the error routes to the planner, not the executor."""
    step = make_step(1, status=RunStatus.FAILED)
    state = make_state(plan=make_plan(step), current_step=0, retry_count=1)
    assert retry_router(state) == PLANNER


def test_retry_router_replans_on_verification_rejection() -> None:
    """A step that ran but was rejected is not FAILED -> replan, since a verbatim
    re-run would reproduce the rejected output."""
    step = make_step(1, status=RunStatus.COMPLETED)
    state = make_state(plan=make_plan(step), current_step=0, retry_count=1)
    assert retry_router(state) == PLANNER


def test_retry_router_responds_when_no_recoverable_step() -> None:
    state = make_state(plan=make_plan(make_step(1)), current_step=5, retry_count=0)
    assert retry_router(state) == RESPONDER


def test_retry_router_responds_when_plan_missing() -> None:
    state = make_state(plan=None, current_step=0, retry_count=0)
    assert retry_router(state) == RESPONDER

"""Tests for the phase-transition node (``agent_langgraph.nodes.phase_transition``).

The node owns *phase policy only*: given an exhausted plan it picks the next
``workflow_phase`` and lifts durable observations into ``findings`` before the
replan swaps out ``plan``. Turning a phase into a destination is
``phase_router``'s job (``test_routing.py``); running two phases end to end is
``tests/e2e/test_agent_flow.py``. Here every policy branch is pinned.

Two invariants matter most and are the easiest to regress:
  - the second phase is *opt-in* (``requires_remediation``) — a single-phase
    task must not silently run twice;
  - only COMPLETED + verified + non-empty steps are trustworthy findings.
"""

from __future__ import annotations

from typing import Any

from agent_langgraph.graph.state import Finding
from agent_langgraph.nodes.phase_transition import PhaseTransitionNode
from common.enums import RunStatus, TaskStatus, WorkflowPhase
from langchain_core.messages import SystemMessage

from tests.support.langgraph import make_plan, make_state, make_step


def _verified_step(step_id: int = 1, output: str = "found something", **overrides: Any):
    """A step the harvester should accept: completed, verified, real output."""
    fields: dict[str, Any] = {
        "status": RunStatus.COMPLETED,
        "verified": True,
        "output": output,
    }
    fields.update(overrides)
    return make_step(step_id, **fields)


# ---------------------------------------------------------------------------
# investigate -> remediate : opt-in follow-up, with findings to act on
# ---------------------------------------------------------------------------


def test_investigate_with_findings_and_opt_in_advances_to_remediate() -> None:
    state = make_state(
        workflow_phase=WorkflowPhase.INVESTIGATE,
        plan=make_plan(_verified_step(1, "bug in foo.py"), requires_remediation=True),
    )
    delta = PhaseTransitionNode(state)

    assert delta["workflow_phase"] is WorkflowPhase.REMEDIATE
    assert delta["status"] is TaskStatus.PLANNING
    # A marker message is added so the phase change is visible in the transcript.
    assert isinstance(delta["messages"][0], SystemMessage)
    assert "remediate" in delta["messages"][0].content


def test_investigate_harvests_only_the_new_findings() -> None:
    """The ``findings`` reducer appends, so the delta must carry only the
    freshly-harvested findings, not the ones already accumulated."""
    prior = Finding(step_id=99, description="earlier", detail="d")
    state = make_state(
        workflow_phase=WorkflowPhase.INVESTIGATE,
        plan=make_plan(_verified_step(1, "new observation"), requires_remediation=True),
        findings=[prior],
    )
    delta = PhaseTransitionNode(state)

    assert len(delta["findings"]) == 1
    assert delta["findings"][0].detail == "new observation"
    assert delta["findings"][0].step_id == 1


# ---------------------------------------------------------------------------
# investigate -> complete : plan satisfied the goal, or nothing to act on
# ---------------------------------------------------------------------------


def test_investigate_without_opt_in_completes_directly() -> None:
    """A plan that already satisfied the goal (requires_remediation False) ends
    the run — this is what keeps ordinary single-phase tasks single-phase."""
    state = make_state(
        workflow_phase=WorkflowPhase.INVESTIGATE,
        plan=make_plan(_verified_step(1), requires_remediation=False),
    )
    delta = PhaseTransitionNode(state)

    assert delta["workflow_phase"] is WorkflowPhase.COMPLETE
    assert delta["status"] is TaskStatus.RESPONDING
    assert "messages" not in delta  # no next planner, so no marker


def test_investigate_opt_in_but_no_findings_completes() -> None:
    """Opt-in remediation with an empty investigation has nothing to act on, so
    there is no point starting a second phase."""
    state = make_state(
        workflow_phase=WorkflowPhase.INVESTIGATE,
        plan=make_plan(
            make_step(1, status=RunStatus.COMPLETED, verified=True, output=""),
            requires_remediation=True,
        ),
        findings=[],
    )
    delta = PhaseTransitionNode(state)

    assert delta["workflow_phase"] is WorkflowPhase.COMPLETE
    assert delta["status"] is TaskStatus.RESPONDING


# ---------------------------------------------------------------------------
# _harvest : only trustworthy steps become findings
# ---------------------------------------------------------------------------


def test_harvest_skips_unverified_incomplete_and_empty_steps() -> None:
    state = make_state(
        workflow_phase=WorkflowPhase.INVESTIGATE,
        plan=make_plan(
            _verified_step(1, "kept"),
            make_step(2, status=RunStatus.COMPLETED, verified=False, output="unverified"),
            make_step(3, status=RunStatus.PENDING, verified=True, output="not run"),
            make_step(4, status=RunStatus.COMPLETED, verified=True, output="   "),
            requires_remediation=True,
        ),
    )
    delta = PhaseTransitionNode(state)

    assert [f.detail for f in delta["findings"]] == ["kept"]
    assert delta["findings"][0].phase is WorkflowPhase.INVESTIGATE
    assert delta["findings"][0].source_tool == "echo_tool"


# ---------------------------------------------------------------------------
# remediate -> complete : follow-up applied, always terminates
# ---------------------------------------------------------------------------


def test_remediate_always_completes() -> None:
    """From remediate the only forward move is complete, regardless of
    ``requires_remediation`` — this is what bounds the number of phases and
    guarantees the graph terminates."""
    state = make_state(
        workflow_phase=WorkflowPhase.REMEDIATE,
        plan=make_plan(_verified_step(1, "applied fix"), requires_remediation=True),
    )
    delta = PhaseTransitionNode(state)

    assert delta["workflow_phase"] is WorkflowPhase.COMPLETE
    assert delta["status"] is TaskStatus.RESPONDING
    assert "messages" not in delta


# ---------------------------------------------------------------------------
# defaults / defensiveness
# ---------------------------------------------------------------------------


def test_missing_phase_defaults_to_investigate() -> None:
    state = make_state(plan=make_plan(_verified_step(1), requires_remediation=True))
    state.pop("workflow_phase", None)
    delta = PhaseTransitionNode(state)
    # Treated as investigate: opt-in + a real finding -> remediate.
    assert delta["workflow_phase"] is WorkflowPhase.REMEDIATE


def test_missing_plan_completes_without_error() -> None:
    state = make_state(plan=None, workflow_phase=WorkflowPhase.INVESTIGATE)
    delta = PhaseTransitionNode(state)
    assert delta["workflow_phase"] is WorkflowPhase.COMPLETE
    assert delta["findings"] == []


def test_retry_count_is_left_untouched() -> None:
    """The retry budget is shared across the whole run, replans included, so the
    transition must not reset or otherwise write it."""
    state = make_state(
        workflow_phase=WorkflowPhase.INVESTIGATE,
        plan=make_plan(_verified_step(1), requires_remediation=True),
        retry_count=2,
    )
    delta = PhaseTransitionNode(state)
    assert "retry_count" not in delta

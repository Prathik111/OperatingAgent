"""Tests for the graph state models (``agent_langgraph.graph.state``).

``PlanStep`` / ``AgentPlan`` are the checkpointed payload. The executor and
verifier update steps via ``model_copy(update=...)`` precisely so they never
mutate a model that a checkpoint may still reference — these tests lock in the
defaults and that copy-on-write behaviour.
"""

from __future__ import annotations

import pytest
from agent_langgraph.graph.state import AgentPlan, Finding, PlanStep
from common.enums import RunStatus, VerificationResult, WorkflowPhase
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------


def test_plan_step_defaults() -> None:
    step = PlanStep(id=1, description="do a thing")
    assert step.tool_name is None
    assert step.arguments == {}
    assert step.verified is False
    assert step.verification is None
    assert step.status is RunStatus.PENDING
    assert step.output is None


def test_plan_step_requires_id_and_description() -> None:
    with pytest.raises(ValidationError):
        PlanStep(description="missing id")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        PlanStep(id=1)  # type: ignore[call-arg]


def test_plan_step_coerces_enum_fields() -> None:
    step = PlanStep(
        id=1,
        description="d",
        status=RunStatus.COMPLETED,
        verification=VerificationResult.VERIFIED,
    )
    assert step.status is RunStatus.COMPLETED
    assert step.verification is VerificationResult.VERIFIED


def test_model_copy_update_is_non_mutating() -> None:
    """This is the pattern both nodes use; the original must be untouched."""
    original = PlanStep(id=1, description="d", tool_name="t")
    updated = original.model_copy(update={"status": RunStatus.COMPLETED, "output": "ok"})
    assert original.status is RunStatus.PENDING
    assert original.output is None
    assert updated.status is RunStatus.COMPLETED
    assert updated.output == "ok"
    assert updated.id == 1 and updated.tool_name == "t"


def test_arguments_default_is_per_instance() -> None:
    a = PlanStep(id=1, description="a")
    b = PlanStep(id=2, description="b")
    a.arguments["k"] = "v"
    assert b.arguments == {}


# ---------------------------------------------------------------------------
# AgentPlan
# ---------------------------------------------------------------------------


def test_agent_plan_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        AgentPlan(summary="s", reasoning="r")  # type: ignore[call-arg]


def test_agent_plan_holds_ordered_steps() -> None:
    plan = AgentPlan(
        summary="s",
        reasoning="r",
        steps=[PlanStep(id=1, description="first"), PlanStep(id=2, description="second")],
    )
    assert [s.id for s in plan.steps] == [1, 2]


def test_agent_plan_step_replacement_is_immutable() -> None:
    """The executor rebuilds the whole plan with one step swapped; the source
    plan and its untouched steps must survive."""
    plan = AgentPlan(
        summary="s", reasoning="r",
        steps=[PlanStep(id=1, description="a"), PlanStep(id=2, description="b")],
    )
    new_steps = list(plan.steps)
    new_steps[0] = new_steps[0].model_copy(update={"status": RunStatus.COMPLETED})
    new_plan = plan.model_copy(update={"steps": new_steps})

    assert plan.steps[0].status is RunStatus.PENDING  # original intact
    assert new_plan.steps[0].status is RunStatus.COMPLETED
    assert new_plan.steps[1] is plan.steps[1]  # untouched step shared, not copied


def test_agent_plan_accepts_empty_step_list() -> None:
    plan = AgentPlan(summary="nothing to do", reasoning="trivial", steps=[])
    assert plan.steps == []


def test_agent_plan_requires_remediation_defaults_false() -> None:
    """The second phase is opt-in: a plan is single-phase unless it explicitly
    asks for follow-up, so ordinary tasks don't silently run twice."""
    plan = AgentPlan(summary="s", reasoning="r", steps=[])
    assert plan.requires_remediation is False


# ---------------------------------------------------------------------------
# Finding — the durable observation carried across a replan
# ---------------------------------------------------------------------------


def test_finding_defaults() -> None:
    finding = Finding(step_id=1, description="what was checked", detail="what was seen")
    assert finding.source_tool is None
    assert finding.phase is WorkflowPhase.INVESTIGATE


def test_finding_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        Finding(step_id=1, description="missing detail")  # type: ignore[call-arg]

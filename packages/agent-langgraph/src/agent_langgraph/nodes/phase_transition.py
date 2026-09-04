from __future__ import annotations

import logging

from agent_langgraph.graph.state import AgentPlan, AgentState, Finding
from common.enums import RunStatus, TaskStatus, WorkflowPhase
from langchain_core.messages import SystemMessage

log = logging.getLogger(__name__)

#: Monotonic phase order. Advancing only ever moves forward, which is what
#: bounds the number of replans and guarantees the graph terminates.
_NEXT_PHASE: dict[WorkflowPhase, WorkflowPhase] = {
    WorkflowPhase.INVESTIGATE: WorkflowPhase.REMEDIATE,
    WorkflowPhase.REMEDIATE: WorkflowPhase.COMPLETE,
    WorkflowPhase.COMPLETE: WorkflowPhase.COMPLETE,
}


def _harvest(plan: AgentPlan | None, phase: WorkflowPhase) -> list[Finding]:
    """Lift durable observations out of an exhausted plan.

    A replan replaces ``plan`` wholesale, so anything the next phase needs has
    to be copied into ``findings`` before that happens. Only steps that
    completed *and* verified with real output are trustworthy enough to carry
    forward.
    """
    if plan is None:
        return []

    harvested: list[Finding] = []
    for step in plan.steps:
        if step.status is not RunStatus.COMPLETED or not step.verified:
            continue
        if step.output is None or not str(step.output).strip():
            continue
        harvested.append(
            Finding(
                step_id=step.id,
                description=step.description,
                detail=str(step.output),
                source_tool=step.tool_name,
                phase=phase,
            )
        )
    return harvested


def PhaseTransitionNode(state: AgentState) -> dict:
    """Advance the workflow phase once the current plan is exhausted.

    Reached when a plan runs out of steps, whether that plan was empty from the
    start or every step verified. It owns *phase policy only* — ``phase_router``
    turns the new phase into a destination:

      - investigate -> remediate : the plan asked for follow-up and findings exist
      - investigate -> complete  : nothing to act on, so the run is done
      - remediate   -> complete  : the follow-up work has been applied

    A second phase is opt-in: the planner sets ``requires_remediation`` when its
    plan only investigates. Defaulting to False keeps single-phase tasks
    single-phase instead of silently running every plan twice.

    Findings are appended (never replaced) so they survive the replan that
    swaps out ``plan``. ``retry_count`` is deliberately left alone: the budget
    is shared across the whole run, replans included.
    """
    phase = state.get("workflow_phase") or WorkflowPhase.INVESTIGATE
    plan: AgentPlan | None = state.get("plan")

    harvested = _harvest(plan, phase)
    total_findings = len(state.get("findings", [])) + len(harvested)

    next_phase = _NEXT_PHASE[phase]

    if phase is WorkflowPhase.INVESTIGATE:
        wants_remediation = bool(plan is not None and plan.requires_remediation)
        # Skip remediation when the plan already satisfied the goal, or when the
        # investigation turned up nothing to act on.
        if not wants_remediation or total_findings == 0:
            next_phase = WorkflowPhase.COMPLETE

    log.info(
        "phase transition %s -> %s (%d new findings, %d total)",
        phase.value, next_phase.value, len(harvested), total_findings,
    )

    delta: dict = {
        "workflow_phase": next_phase,
        # Appended by the reducer — only the new ones go here.
        "findings": harvested,
        "status": (
            TaskStatus.RESPONDING
            if next_phase is WorkflowPhase.COMPLETE
            else TaskStatus.PLANNING
        ),
    }

    # Give the next planner an explicit note that the phase moved on. The full
    # findings list is injected by the planner itself; this is just the marker
    # that survives in the conversation transcript.
    if next_phase is not WorkflowPhase.COMPLETE:
        delta["messages"] = [SystemMessage(content=(
            f"[phase] {phase.value} complete with {total_findings} finding(s). "
            f"Now planning the {next_phase.value} phase."
        ))]

    return delta

from common.enums import WorkflowPhase

from .constants import (
    ERROR_HANDLER,
    EXECUTOR,
    MAX_RETRIES,
    PHASE_TRANSITION,
    PLANNER,
    RESPONDER,
    NodeType,
)
from .state import AgentState


def should_execute(state: AgentState) -> NodeType:
    """
    Determines whether the plan should be executed based on the current state.

    A planner-produced plan either has work for the executor or is complete and
    can go directly to the responder. Phase advancement is only considered
    after the verifier exhausts an executed plan.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        NodeType: The next node to transition to based on the execution decision.
    """
    plan = state.get("plan")

    if plan is not None and plan.steps and state.get("current_step", 0) < len(plan.steps):
        return EXECUTOR
    return RESPONDER

def verification_router(state: AgentState) -> NodeType:
    """
    Routes after the verifier runs.

    The verifier advances ``current_step`` only on a passing verdict, so:
      - pass + more steps remain -> execute the next step
      - pass + plan exhausted     -> phase transition (which may start a new
                                     phase, or finish the run)
      - fail                      -> error handler decides retry vs. give up

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        NodeType: The next node to transition to based on the verification result.
    """
    if not state.get("verification_success", False):
        return ERROR_HANDLER

    plan = state.get("plan")
    if plan is not None and state.get("current_step", 0) < len(plan.steps):
        return EXECUTOR
    return PHASE_TRANSITION


def phase_router(state: AgentState) -> NodeType:
    """
    Routes after the phase transition node has advanced ``workflow_phase``.

    Pure: the node owns the phase policy, this only turns the resulting phase
    into a destination.

      - complete -> respond (no further phases)
      - anything else -> planner, to build that phase's plan

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        NodeType: The next node to transition to for the new phase.
    """
    if state.get("workflow_phase") is WorkflowPhase.COMPLETE:
        return RESPONDER
    return PLANNER


def retry_router(state: AgentState) -> NodeType:
    """
    Routes after the error handler runs. Reached only on a failure, so the
    decision is replan-vs-give-up:

      - budget spent      -> respond (report the failure honestly)
      - recoverable step  -> replan: hand back to the planner to modify the
                             approach around the failure.

    Every failure replans rather than re-running the same step: a tool error or
    a rejected result would recur on a verbatim retry, and genuinely transient
    faults have already been retried inside the executor. The error handler
    leaves a diagnostic on ``messages`` so the replan is informed by what went
    wrong.

    Pure: it reads state and returns a destination; it does not mutate state.
    ``retry_count`` is incremented upstream (executor on a tool failure,
    verifier on a rejection) and is preserved across the replan, so counting
    here would double-count and a run that keeps failing still gives up after
    the configured budget instead of replanning forever. The budget itself is
    ``state["max_replans"]``, seeded by the orchestrator from
    ``execution.max_replans`` (env ``AGENT_MAX_REPLANS``); states that predate
    the field fall back to the module default.

    Args:
        state (AgentState): The agent state to inspect.

    Returns:
        NodeType: The node to transition to after an error.
    """
    budget = state.get("max_replans") or MAX_RETRIES
    if state.get("retry_count", 0) >= budget:
        return RESPONDER
    if str(state.get("last_error") or "").startswith("human rejected "):
        return RESPONDER
    if str(state.get("last_error") or "").startswith("blocked "):
        # A policy block is deterministic: replanning would propose the same
        # refused call until the budget runs out. Report it instead.
        return RESPONDER

    plan = state.get("plan")
    if plan is not None and state.get("current_step", 0) < len(plan.steps):
        return PLANNER

    return RESPONDER

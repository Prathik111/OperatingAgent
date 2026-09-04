from __future__ import annotations

import logging

from agent_langgraph.graph.state import AgentPlan, AgentState
from common.enums import RunStatus, TaskStatus
from langchain_core.messages import SystemMessage

log = logging.getLogger(__name__)


def ErrorHandlerNode(state: AgentState) -> dict:
    """Prepare recovery context after a step fails or is rejected.

    This node owns *policy and preparation*, not routing — ``retry_router``
    decides retry-vs-replan-vs-give-up from the state this returns. It:

      - leaves ``retry_count`` untouched (already incremented by the executor
        on a tool failure, or by the verifier on a rejection),
      - leaves the failed step's status intact so ``retry_router`` can tell an
        execution failure (retry the same step) from a verification rejection
        (replan around it),
      - appends a diagnostic note to ``messages`` so a replan is informed by
        what actually went wrong.
    """
    plan: AgentPlan | None = state.get("plan")
    index = state.get("current_step", 0)
    reason = state.get("last_error") or "unknown error"

    if plan is None or index >= len(plan.steps):
        # Nothing actionable to recover; let the router send us onward.
        log.warning("error handler reached with no recoverable step: %s", reason)
        return {"status": TaskStatus.FAILED, "last_error": reason}

    step = plan.steps[index]
    mode = "execution failure" if step.status is RunStatus.FAILED else "verification rejection"

    log.info(
        "recovering step %d (%s): %s [attempt %d]",
        step.id, mode, reason, state.get("retry_count", 0),
    )

    diagnostic = SystemMessage(content=(
        f"[recovery] Step {step.id} ({step.description!r}) failed due to a "
        f"{mode}: {reason}. If retrying is unlikely to help, adjust the "
        f"approach for this step."
    ))

    return {
        "messages": [diagnostic],
        "status": TaskStatus.EXECUTING,
    }

from __future__ import annotations

import logging

from agent_langgraph.graph.state import AgentPlan, AgentState, PlanStep
from agent_langgraph.runtime.context import AgentContext
from common.enums import RunStatus, TaskStatus, VerificationResult
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class _Verdict(BaseModel):
    """Structured judgement returned by the LLM verifier."""

    success: bool = Field(description="True if the step output satisfies its intent.")
    reason: str = Field(description="Short justification for the verdict.")


def _with_step(plan: AgentPlan, index: int, **updates) -> AgentPlan:
    """Immutable step update — mirrors the executor so checkpoints stay intact."""
    steps = list(plan.steps)
    steps[index] = steps[index].model_copy(update=updates)
    return plan.model_copy(update={"steps": steps})


def _deterministic_verdict(step: PlanStep) -> _Verdict | None:
    """Cheap, conclusive checks that avoid an LLM round-trip.

    Returns a verdict when the outcome is certain, else None to defer to the
    model. A failed tool call or empty output needs no judgement.
    """
    if step.status is RunStatus.FAILED:
        return _Verdict(success=False, reason=step.output or "step failed during execution")
    if step.output is None or not str(step.output).strip():
        return _Verdict(success=False, reason="step produced no output")
    return None


async def _llm_verdict(ctx: AgentContext, goal: str, step: PlanStep) -> _Verdict:
    """Ask the model whether the step output achieves its intent."""
    system_prompt = ctx.prompt_manager.verifier()
    model = ctx.model_provider.get_model().with_structured_output(_Verdict)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Overall goal:\n{goal}\n\n"
            f"Step {step.id}: {step.description}\n"
            f"Tool used: {step.tool_name or 'none (reasoning step)'}\n"
            f"Output produced:\n{step.output}\n\n"
            "Decide whether this step achieved its intended outcome."
        )),
    ]

    verdict = await model.ainvoke(messages)
    return verdict if isinstance(verdict, _Verdict) else _Verdict.model_validate(verdict)


async def VerifierNode(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Judge the just-executed step and, on success, advance the plan pointer.

    The pointer advance here is the commit: the executor deliberately leaves
    ``current_step`` untouched so a rejected step re-runs the same index. On a
    passing verdict we mark the step verified and move to the next one; on a
    failing verdict we leave the pointer put and hand off to the error handler.
    """
    ctx = runtime.context
    plan: AgentPlan | None = state.get("plan")
    index = state.get("current_step", 0)

    # Nothing to verify (defensive; the executor guards this too).
    if plan is None or index >= len(plan.steps):
        return {"status": TaskStatus.RESPONDING}

    step = plan.steps[index]

    # 1. Deterministic layer first — free, and conclusive when it fires.
    verdict = _deterministic_verdict(step)

    # 2. LLM layer only when needed and enabled. If the verifier itself is
    #    unavailable (e.g. prompt file missing during build-out), fail closed
    #    rather than silently passing unverified work.
    if verdict is None:
        if ctx.config.behaviour.require_verification:
            try:
                verdict = await _llm_verdict(ctx, state.get("goal", ""), step)
            except Exception as exc:  # noqa: BLE001 - verification fails closed
                log.warning("LLM verification failed for step %d: %s", index, exc)
                verdict = _Verdict(success=False, reason=f"verifier unavailable: {exc}")
        else:
            verdict = _Verdict(success=True, reason="verification disabled")

    # --- Accept: mark verified and advance. This advance is the commit. -----
    if verdict.success:
        next_index = index + 1
        done = next_index >= len(plan.steps)
        return {
            "plan": _with_step(plan, index,
                               verified=True,
                               verification=VerificationResult.VERIFIED,
                               status=RunStatus.COMPLETED),
            "current_step": next_index,
            "verification_success": True,
            "verification_reason": verdict.reason,
            "last_error": None,
            "status": TaskStatus.RESPONDING if done else TaskStatus.EXECUTING,
        }

    # --- Reject: leave the pointer; let the error handler decide next. -------
    already_failed = step.status is RunStatus.FAILED  # executor already counted this
    return {
        "plan": _with_step(plan, index,
                            verified=False,
                            verification=VerificationResult.NOT_VERIFIED,
                            # A rejected completed step is no longer terminal;
                            # keep FAILED for execution failures so recovery can
                            # distinguish the two cases.
                            status=RunStatus.FAILED if already_failed else RunStatus.PENDING),
        "verification_success": False,
        "verification_reason": verdict.reason,
        "last_error": verdict.reason,
        "retry_count": state.get("retry_count", 0) + (0 if already_failed else 1),
        "status": TaskStatus.EXECUTING,
    }

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from agent_langgraph.graph.state import AgentPlan, AgentState, PlanStep
from agent_langgraph.runtime.context import AgentContext
from common.approvals import ApprovalRequest
from common.enums import RiskLevel, RunStatus, TaskStatus
from common.events import AgentEvent
from common.tools import ToolCallRequest, ToolCallResult
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8_000

# Risk levels at or above the configured threshold require a human gate.
_RISK_ORDER = {RiskLevel.SAFE: 0, RiskLevel.REVIEW: 1, RiskLevel.BLOCKED: 2}


class ToolFailure(Exception):
    """Raised when a step cannot be completed.

    ``retryable`` decides whether the inner loop retries the same call
    (transient faults such as a timeout) or gives up immediately
    (deterministic faults such as a policy block, which would fail
    identically on every retry).
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _with_step(plan: AgentPlan, index: int, **updates: Any) -> AgentPlan:
    """Return a new plan with ``steps[index]`` updated; never mutates ``plan``.

    Mutating a model that lives inside checkpointed state can corrupt earlier
    checkpoints (the checkpointer may hold the same reference), so every step
    change goes through an immutable copy.
    """
    steps = list(plan.steps)
    steps[index] = steps[index].model_copy(update=updates)
    return plan.model_copy(update={"steps": steps})


def _stringify(output: Any) -> str:
    text = output if isinstance(output, str) else repr(output)
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n... [{omitted} chars omitted] ...\n{text[-half:]}"


def _needs_approval(ctx: AgentContext, step: PlanStep) -> RiskLevel:
    """Classify a step's risk deterministically.

    Delegates to the shared RiskClassifier (``classify(call) -> RiskLevel``).
    Any unexpected failure degrades to SAFE rather than crashing execution.
    """
    try:
        return ctx.risk_classifier.classify(
            ToolCallRequest(tool_name=step.tool_name or "", arguments=step.arguments)
        )
    except Exception as exc:  # noqa: BLE001 - injected classifier boundary
        log.warning("risk classification failed for %s: %s", step.tool_name, exc)
        return RiskLevel.SAFE


async def ExecutorNode(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Execute the current plan step and return only the changed state.

    The node runs exactly one step per invocation; the loop is the graph
    cycle executor -> verifier -> executor, so every step is a checkpoint
    boundary. The step pointer (``current_step``) is advanced by the verifier
    on success, not here, so a retry naturally re-runs the same index.
    """
    ctx = runtime.context
    plan: AgentPlan | None = state.get("plan")
    index = state.get("current_step", 0)

    # --- Guards ---------------------------------------------------------
    if plan is None:
        return {"status": TaskStatus.FAILED,
                "last_error": "executor reached with no plan"}

    if index >= len(plan.steps):
        return {"status": TaskStatus.RESPONDING}

    if index >= ctx.config.execution.max_iterations:
        return {"status": TaskStatus.FAILED,
                "last_error": f"iteration budget ({ctx.config.execution.max_iterations}) exhausted"}

    step = plan.steps[index]

    if not step.tool_name:
        # A reasoning-only step: nothing to invoke, record and hand to verifier.
        return {
            "plan": _with_step(plan, index,
                               status=RunStatus.COMPLETED,
                               output=step.description),
            "status": TaskStatus.VERIFYING,
        }

    # --- Human gate for risky tools -------------------------------------
    risk = _needs_approval(ctx, step)
    threshold = _RISK_ORDER.get(RiskLevel(ctx.config.behaviour.risk_threshold), 1)
    if ctx.config.behaviour.require_human_approval and _RISK_ORDER[risk] >= threshold:
        if not ctx.config.execution.enable_interrupts:
            approved = False
            reason = "approval required but interrupts are disabled"
        elif ctx.approval_handler is not None:
            approved = await ctx.approval_handler.request_approval(
                ApprovalRequest(
                    id=_approval_id(
                        ctx.task_id,
                        state.get("workflow_phase"),
                        step.id,
                        step.tool_name or "",
                        step.arguments,
                    ),
                    task_id=ctx.task_id,
                    tool_name=step.tool_name or "",
                    arguments=step.arguments,
                    risk_level=risk,
                    description=step.description,
                )
            )
            reason = "human rejected the tool call"
        else:
            decision = interrupt({
                "kind": "tool_approval",
                "step": step.id,
                "tool": step.tool_name,
                "arguments": step.arguments,
                "risk": risk.value,
                "description": step.description,
            })
            approved = isinstance(decision, dict) and bool(decision.get("approved"))
            reason = (
                decision.get("reason", "no reason given")
                if isinstance(decision, dict)
                else "rejected"
            )

        if not approved:
            return {
                "plan": _with_step(plan, index,
                                   status=RunStatus.FAILED,
                                   output=f"human rejected {step.tool_name}: {reason}"),
                "last_error": f"human rejected {step.tool_name}: {reason}",
                "retry_count": state.get("retry_count", 0) + 1,
                "status": TaskStatus.EXECUTING,
            }

    # --- Invoke with retry ----------------------------------------------
    try:
        output = await _invoke_with_retry(
            ctx, step, index, state.get("workflow_phase")
        )
    except ToolFailure as exc:
        log.warning("step %d failed: %s (retryable=%s)", index, exc, exc.retryable)
        return {
            "plan": _with_step(plan, index,
                               status=RunStatus.FAILED,
                               output=str(exc)),
            "last_error": str(exc),
            "retry_count": state.get("retry_count", 0) + 1,
            "status": TaskStatus.EXECUTING,
        }

    # --- Success: delta only. Pointer advanced by the verifier. ---------
    return {
        "plan": _with_step(plan, index,
                           status=RunStatus.COMPLETED,
                           output=output),
        "messages": [AIMessage(content=f"[step {step.id}] {step.description}\n{output}")],
        "last_error": None,
        "retry_count": 0,
        "status": TaskStatus.VERIFYING,
    }


async def _invoke_with_retry(
    ctx: AgentContext, step: PlanStep, index: int, phase: Any = None
) -> str:
    """Call the tool, retrying only transient failures with backoff."""
    attempts = ctx.config.execution.retry_attempts + 1

    for attempt in range(attempts):
        try:
            return await _invoke_once(ctx, step, phase)
        except ToolFailure as exc:
            if not exc.retryable or attempt == attempts - 1:
                raise
            backoff = 2 ** attempt
            log.info("step %d attempt %d failed, retrying in %ss", index, attempt + 1, backoff)
            await asyncio.sleep(backoff)

    raise AssertionError("unreachable")


async def _invoke_once(ctx: AgentContext, step: PlanStep, phase: Any = None) -> str:
    """Resolve and call one tool through the registry with a timeout.

    Wrapped in a ``tool``-typed observation: MCP calls bypass LangChain, so the
    Langfuse CallbackHandler cannot see them: without this span the trace would
    show a step executing with no record of the tool that did the work.
    """
    tool_name = step.tool_name
    if tool_name is None:
        raise ToolFailure("executor received a step without a tool", retryable=False)

    call_id = _tool_call_id(
        ctx.task_id, phase, step.id, tool_name, step.arguments
    )
    completed = (ctx.completed_tool_calls or {}).get(call_id)
    if completed is not None:
        log.info("reusing completed tool call %s after resume", call_id)
        return completed

    await _emit_event(
        ctx,
        AgentEvent(
            type="tool_started",
            payload={
                "call_id": call_id,
                "step_id": step.id,
                "tool": tool_name,
                "arguments": step.arguments,
            },
        ),
    )

    with ctx.tracer.observation(
        tool_name,
        as_type="tool",
        input=step.arguments,
        metadata={"step_id": step.id, "description": step.description},
    ) as span:
        try:
            if ctx.workspace:
                call = ctx.tool_registry.call_by_name(
                    tool_name,
                    step.arguments,
                    workspace=ctx.workspace,
                )
            else:
                call = ctx.tool_registry.call_by_name(tool_name, step.arguments)
            result: ToolCallResult = await asyncio.wait_for(
                call,
                timeout=ctx.config.execution.timeout_seconds,
            )
        except TimeoutError as exc:
            message = f"{tool_name} exceeded {ctx.config.execution.timeout_seconds}s"
            span.update(level="ERROR", status_message=message)
            await _emit_tool_finished(ctx, call_id, step, tool_name, False, error=message)
            raise ToolFailure(message, retryable=True) from exc
        except Exception as exc:  # transport / adapter errors are usually transient
            message = f"{tool_name} raised: {exc}"
            span.update(level="ERROR", status_message=message)
            await _emit_tool_finished(ctx, call_id, step, tool_name, False, error=message)
            raise ToolFailure(message, retryable=True) from exc

        if not result.success:
            # The tool ran and reported a business failure; re-running verbatim
            # will not help, so route to replanning rather than retrying.
            message = result.error or f"{tool_name} reported failure"
            span.update(level="ERROR", status_message=message)
            await _emit_tool_finished(ctx, call_id, step, tool_name, False, error=message)
            raise ToolFailure(message, retryable=False)

        output = _stringify(result.output)
        span.update(output=output)
        await _emit_tool_finished(ctx, call_id, step, tool_name, True, output=output)
        return output


async def _emit_tool_finished(
    ctx: AgentContext,
    call_id: str,
    step: PlanStep,
    tool_name: str,
    success: bool,
    *,
    output: str | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "call_id": call_id,
        "step_id": step.id,
        "tool": tool_name,
        "success": success,
    }
    if output is not None:
        payload["output"] = output
    if error is not None:
        payload["error"] = error
    await _emit_event(ctx, AgentEvent(type="tool_finished", payload=payload))


async def _emit_event(ctx: AgentContext, event: AgentEvent) -> None:
    if ctx.event_sink is None:
        return
    try:
        outcome = ctx.event_sink(event)
        if outcome is not None and hasattr(outcome, "__await__"):
            await outcome
    except Exception as exc:  # noqa: BLE001 - event persistence is best effort
        log.warning("could not persist %s event: %s", event.type, exc)


def _phase_value(phase: Any) -> str:
    return getattr(phase, "value", None) or "investigate"


def _tool_call_id(
    task_id: str, phase: Any, step_id: int, tool_name: str, arguments: dict
) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"operating-agent:tool:{task_id}:{_phase_value(phase)}:{step_id}:{_step_fingerprint(tool_name, arguments)}",
        )
    )


def _approval_id(
    task_id: str,
    phase: Any,
    step_id: int,
    tool_name: str,
    arguments: dict,
) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"operating-agent:approval:{task_id}:{_phase_value(phase)}:{step_id}:{_step_fingerprint(tool_name, arguments)}",
        )
    )


def _step_fingerprint(tool_name: str, arguments: dict) -> str:
    try:
        canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        canonical = repr(arguments)
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()[:24]

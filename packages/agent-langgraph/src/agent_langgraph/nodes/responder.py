from __future__ import annotations

import logging
from typing import Any

from agent_langgraph.graph.state import AgentPlan, AgentState, Finding
from agent_langgraph.runtime.context import AgentContext
from common.enums import RunStatus, TaskStatus
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

log = logging.getLogger(__name__)

#: Per-finding detail cap when summarising, mirroring the planner's cap.
_MAX_FINDING_DETAIL = 1_500

#: How many trailing conversation messages the direct-answer path may quote.
#: Bounded so a long chat cannot bloat a request that needs no tools at all.
_MAX_HISTORY_MESSAGES = 10


def _succeeded(plan: AgentPlan | None, last_error: str | None) -> bool:
    """A run succeeds when nothing is outstanding: no error, and every step
    completed and verified. An empty/trivial plan counts as success."""
    if last_error is not None:
        return False
    if plan is None:
        return True
    return all(
        s.status is RunStatus.COMPLETED and s.verified for s in plan.steps
    )


def _build_findings_block(findings: list[Finding]) -> str:
    """Render accumulated findings, grouped by the phase that produced them.

    The final plan only covers the last phase, so without this the answer would
    silently drop everything the investigation discovered.
    """
    if not findings:
        return ""

    lines = []
    current_phase = None
    for finding in findings:
        phase_value = getattr(finding.phase, "value", str(finding.phase))
        if phase_value != current_phase:
            current_phase = phase_value
            lines.append(f"\n[{phase_value} phase]")
        detail = finding.detail
        if len(detail) > _MAX_FINDING_DETAIL:
            detail = f"{detail[:_MAX_FINDING_DETAIL]}... [truncated]"
        source = f" (via {finding.source_tool})" if finding.source_tool else ""
        lines.append(f"- {finding.description}{source}: {detail}")
    return "\n".join(lines)


def _build_transcript(plan: AgentPlan | None) -> str:
    """Render executed steps and their outputs for the model to summarise."""
    if plan is None or not plan.steps:
        return "(no steps were executed)"

    lines = []
    for step in plan.steps:
        marker = "ok" if (step.status is RunStatus.COMPLETED and step.verified) else step.status.value
        lines.append(f"[{marker}] step {step.id}: {step.description}")
        if step.output:
            lines.append(f"      -> {step.output}")
    return "\n".join(lines)


def _recent_history(messages: Any) -> str:
    """Render the trailing conversation for a direct answer. "" when empty.

    LangChain messages carry ``type``/``content``; anything else degrades to
    its class name and string form rather than breaking the request.
    """
    if not isinstance(messages, list) or not messages:
        return ""
    lines = []
    for message in messages[-_MAX_HISTORY_MESSAGES:]:
        role = getattr(message, "type", None) or type(message).__name__
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        text = str(content).strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _fallback_summary(
    plan: AgentPlan | None,
    succeeded: bool,
    last_error: str | None,
    findings: list[Finding] | None = None,
) -> str:
    """Deterministic answer used when the LLM/prompt is unavailable, so the
    graph always terminates with something meaningful."""
    if succeeded:
        count = len(findings or [])
        outputs = [s.output for s in (plan.steps if plan else []) if s.output]
        prefix = f"Completed with {count} finding(s). " if count else "Completed. "
        if outputs:
            return prefix + " ".join(outputs)
        return prefix.strip() or "Completed the requested work."
    return f"Could not complete the task: {last_error or 'unknown error'}."


async def ResponderNode(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Synthesise the final user-facing answer and mark the run terminal.

    Terminal for every path through the graph: success (all steps verified),
    a trivial plan, or give-up after the retry budget is spent. It never
    routes onward — the builder edges it straight to END.

    In a multi-phase run the current ``plan`` is only the *last* phase, so the
    accumulated ``findings`` are folded in as well — otherwise an investigation's
    results would never reach the answer.
    """
    ctx = runtime.context
    plan: AgentPlan | None = state.get("plan")
    last_error = state.get("last_error")
    succeeded = _succeeded(plan, last_error)

    goal = state.get("goal", "")
    findings = state.get("findings", [])
    transcript = _build_transcript(plan)
    findings_block = _build_findings_block(findings)

    # Synthesise with the model; fall back to a deterministic summary if the
    # responder prompt or provider isn't available.
    try:
        system_prompt = ctx.prompt_manager.responder()
        model = ctx.model_provider.get_model()
        direct_answer = succeeded and not (plan and plan.steps) and not findings
        if direct_answer:
            # An empty plan is intentional for greetings, factual questions, and
            # other requests that need no external action. Ask the responder to
            # answer the goal itself instead of turning the absence of tools into
            # a misleading execution report. The recent conversation goes along:
            # without it a contextual follow-up ("what about the second one?")
            # has nothing to refer to.
            history = _recent_history(state.get("messages", []))
            request = (
                f"Original user request:\n{goal}\n\n"
                + (f"Recent conversation:\n{history}\n\n" if history else "")
                + "Answer the user's request directly and concisely. No external "
                "actions were needed, so do not mention planning, steps, tools, "
                "or an execution transcript."
            )
        else:
            outcome = "succeeded" if succeeded else "failed"
            context_block = (
                f"Original goal:\n{goal}\n\n"
                f"The run {outcome}. Final-phase execution transcript:\n{transcript}\n"
            )
            if findings_block:
                context_block += (
                    f"\nObservations accumulated across all phases:{findings_block}\n"
                )
            request = context_block + "\n" + (
                "Write a concise final answer for the user based on the results "
                "above. Report what was found and what was done about it."
                if succeeded
                else
                f"Explain honestly what was attempted and why it did not complete "
                f"(reason: {last_error}). Do not fabricate success."
            )
        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=request),
        ])
        answer = response.content if isinstance(response.content, str) else str(response.content)
        if not answer.strip():
            answer = _fallback_summary(plan, succeeded, last_error, findings)
    except Exception as exc:  # noqa: BLE001 - terminal fallback must always run
        log.warning("responder synthesis failed, using fallback summary: %s", exc)
        answer = _fallback_summary(plan, succeeded, last_error, findings)

    return {
        "messages": [AIMessage(content=answer)],
        "status": TaskStatus.COMPLETED if succeeded else TaskStatus.FAILED,
    }

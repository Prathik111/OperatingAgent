from __future__ import annotations

import logging

from agent_langgraph.graph.state import AgentPlan, AgentState, Finding
from agent_langgraph.runtime.context import AgentContext
from common.enums import TaskStatus, WorkflowPhase
from common.exceptions import PlanningException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

log = logging.getLogger(__name__)

#: Per-finding detail cap when building the remediation prompt, so a large
#: investigation cannot blow the model's context window.
_MAX_FINDING_DETAIL = 1_500


def _format_findings(findings: list[Finding]) -> str:
    """Render accumulated findings for the planner prompt."""
    lines = []
    for n, finding in enumerate(findings, start=1):
        detail = finding.detail
        if len(detail) > _MAX_FINDING_DETAIL:
            detail = f"{detail[:_MAX_FINDING_DETAIL]}... [truncated]"
        source = f" (via {finding.source_tool})" if finding.source_tool else ""
        lines.append(f"{n}. {finding.description}{source}\n   {detail}")
    return "\n".join(lines)


def _phase_instruction(phase: WorkflowPhase, findings: list[Finding]) -> str:
    """Phase-specific planning instruction appended to the system prompt.

    This is what makes a single planner serve both phases: the same goal yields
    a read-only investigation plan first, then a remediation plan built from
    what that investigation actually found.
    """
    if phase is WorkflowPhase.REMEDIATE:
        if not findings:
            # Defensive: the phase transition skips remediation when nothing was
            # found, so this should not be reachable.
            return (
                "\n\nCurrent phase: REMEDIATE, but no findings were recorded. "
                "Produce the smallest plan that addresses the goal directly."
            )
        return (
            "\n\nCurrent phase: REMEDIATE."
            "\nThe investigation phase recorded the findings below. Produce a plan "
            "that acts on them — the fixes, changes, or follow-up work they call "
            "for. Reference the specific findings your steps address, and do not "
            "create a step for a finding that needs no action. Do not re-run the "
            "investigation."
            f"\n\nFindings from investigation:\n{_format_findings(findings)}"
        )

    return (
        "\n\nCurrent phase: INVESTIGATE."
        "\nIf the goal can be satisfied outright, plan it directly and set "
        "requires_remediation to false."
        "\nIf the goal instead asks you to find something and then act on it "
        "(e.g. 'find and fix bugs', 'check for issues and resolve them'), plan "
        "ONLY the investigation now and set requires_remediation to true: list "
        "and read files, search, inspect status, run tests or checks. In that "
        "case do NOT plan any step that writes, edits, deletes, moves or "
        "otherwise modifies state — the follow-up plan is built separately once "
        "the findings are in. Each investigation step should produce an "
        "observation."
    )


async def _available_tools_hint(runtime: Runtime[AgentContext]) -> str:
    """Describe the tools the planner may reference, defensively.

    The planner produces structured output, so tools are *described* (not
    bound) — that lets it fill in ``PlanStep.tool_name`` with real names. If
    the tool registry / MCP adapter isn't wired yet, planning still proceeds
    without a hint rather than crashing.
    """
    try:
        tools = await runtime.context.tool_registry.list_tools()
    except Exception as exc:  # noqa: BLE001 - pluggable tool registry boundary
        log.warning("could not list tools for planning: %s", exc)
        return ""

    if not tools:
        return ""

    lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    return f"\n\nAvailable tools (reference these by name in tool_name):\n{lines}"


async def planner_function(
    goal: str,
    messages: list,
    runtime: Runtime[AgentContext],
    phase: WorkflowPhase = WorkflowPhase.INVESTIGATE,
    findings: list[Finding] | None = None,
) -> AgentPlan:
    """
    Generate a structured plan for ``goal`` using the configured provider.

    Args:
        goal (str): The goal for which to generate a plan.
        messages (list): Prior conversation messages.
        runtime (Runtime[AgentContext]): Runtime carrying the shared deps.
        phase (WorkflowPhase): Which phase to plan for.
        findings (list[Finding] | None): Observations from earlier phases.

    Returns:
        AgentPlan: The validated structured plan.
    """
    ctx = runtime.context

    model = ctx.model_provider.get_model().with_structured_output(
        AgentPlan, method="json_schema"
    )

    system_prompt = (
        ctx.prompt_manager.planner()
        + _phase_instruction(phase, findings or [])
        + await _available_tools_hint(runtime)
    )
    full_messages = [
        SystemMessage(content=system_prompt),
        *messages,
        HumanMessage(content=f"Generate a plan for the goal: {goal}"),
    ]

    try:
        plan = await model.ainvoke(full_messages)
    except Exception as exc:
        log.error("planning failed with provider %s: %s", ctx.config.llm.provider, exc)
        raise PlanningException(
            f"planner failed with provider {ctx.config.llm.provider}: {exc}"
        ) from exc

    log.info(
        "plan generated with provider %s for phase %s",
        ctx.config.llm.provider, phase.value,
    )
    return plan if isinstance(plan, AgentPlan) else AgentPlan.model_validate(plan)


async def PlannerNode(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """
    Plan the work for the current phase and reset the step pointer.

    Runs once per phase: first for the investigation, then again after the phase
    transition for the remediation, using the findings it accumulated. Neither
    ``findings`` nor ``retry_count`` is touched here — findings must survive the
    replan, and the retry budget is shared across the whole run.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict: State delta with the generated plan.
    """
    goal = state.get("goal")
    if not goal:
        raise PlanningException("planner invoked without a goal")

    # First entry establishes the phase; later entries come from the transition.
    phase = state.get("workflow_phase") or WorkflowPhase.INVESTIGATE
    findings = state.get("findings", [])

    plan = await planner_function(
        goal,
        state.get("messages", []),
        runtime,
        phase=phase,
        findings=findings,
    )

    return {
        "plan": plan,
        "current_step": 0,
        "workflow_phase": phase,
        "status": TaskStatus.PLANNING,
    }

"""Planner - one-shot goal decomposition into an executable Plan.

Structured output via a dedicated `create_plan` tool call (reliable across
providers), with the available tool catalog embedded in the prompt text so
the model picks real tool names. The plan is validated against the catalog
and persisted via this package's own TaskRepository (never another package's
repository).

On invalid output the plan is retried once, then PlanningError is raised -
the agent maps that to a FAILED run with a visible reason.
"""

from __future__ import annotations

import json
import re
import uuid

from ..llm import LLMClient, LLMResponse, ToolCall
from ..repository import TaskRepository
from ..types import AgentTask, Plan, PlanStep, StepKind, ToolInfo

_PLAN_TOOL_NAME = "create_plan"
_PLAN_TOOL_DESCRIPTION = (
    "Create a step-by-step plan to achieve the user's goal. Each step must be "
    "either kind='tool' (executed with a real tool from the catalog; give a "
    "check descriptor when the step result is objectively checkable) or "
    "kind='analysis' (pure reasoning/answer steps with no tool; these are NOT "
    "verifiable and will be trusted on output). "
    "Check descriptor rules: only terminal-like tools (run_command, sh, bash) "
    "support exit_code=<n>; file tools must use file_exists=<path>, "
    "file_absent=<path>, or dir_exists=<path> - never exit_code for them. "
    "Omit the check for steps whose effect cannot be stated as one of these."
)
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "kind": {"type": "string", "enum": ["tool", "analysis"]},
                    "tool_name": {"type": ["string", "null"]},
                    "check": {"type": ["string", "null"], "description":
                               "exit_code=<n> only for terminal-like tools; "
                               "file_exists=<path> | file_absent=<path> | "
                               "dir_exists=<path> for file tools"},
                },
                "required": ["description", "kind"],
            },
        }
    },
    "required": ["steps"],
}


class PlanningError(RuntimeError):
    pass


def _plan_tool() -> ToolInfo:
    from ..types import ToolSchema

    return ToolInfo(
        name=_PLAN_TOOL_NAME,
        description=_PLAN_TOOL_DESCRIPTION,
        schema=ToolSchema(input_schema=_PLAN_SCHEMA, output_schema={}),
    )


def tool_catalog_text(tools: list[ToolInfo]) -> str:
    lines = ["Available tools:"]
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


class Planner:
    def __init__(self, llm: LLMClient, repository: TaskRepository) -> None:
        self.llm = llm
        self.repository = repository

    async def plan(
        self,
        task: AgentTask,
        tools: list[ToolInfo],
        *,
        extra_context: str = "",
    ) -> Plan:
        last_error: str = ""
        for attempt in (1, 2):
            messages = _plan_messages(task, tools, extra_context, last_error)
            try:
                response = await self.llm.complete(messages, tools=[_plan_tool()])
                plan = self._parse_plan(task.id, response, tools)
            except PlanningError as e:
                last_error = str(e)
                continue
            await self.repository.save_plan(plan)
            await self.repository.save_task(task)
            return plan
        raise PlanningError(f"planner failed to produce a valid plan: {last_error}")

    def _parse_plan(self, task_id: str, response: LLMResponse, tools: list[ToolInfo]) -> Plan:
        if not response.wants_tool_call:
            text = (response.text or "").strip()
            if not text:
                raise PlanningError("empty LLM response")
            data = _extract_json(text)
            if data is None:
                raise PlanningError("plan output was not structured JSON")
        else:
            call: ToolCall = response.tool_calls[0]  # type: ignore[index]
            if call.name != _PLAN_TOOL_NAME:
                raise PlanningError(f"unexpected tool call {call.name!r} during planning")
            data = call.arguments
        return build_plan_from_data(task_id, data, tools)


def build_plan_from_data(task_id: str, data: dict, tools: list[ToolInfo]) -> Plan:
    """Shared plan construction/validation (used by planner and reflector)."""
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanningError("plan has no steps")

    tool_names = {t.name for t in tools}
    steps: list[PlanStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            raise PlanningError("malformed step entry")
        desc = str(raw.get("description", "")).strip()
        if not desc:
            raise PlanningError("step missing description")
        kind = str(raw.get("kind", "tool")).lower()
        if kind == "tool":
            tool_name = raw.get("tool_name")
            if tool_name not in tool_names:
                raise PlanningError(
                    f"step {desc[:40]!r} names unknown tool {tool_name!r}"
                )
            check = raw.get("check") or None
            k = StepKind.TOOL
        elif kind == "analysis":
            if raw.get("tool_name"):
                raise PlanningError(
                    f"analysis step {desc[:40]!r} must not name a tool"
                )
            tool_name = None
            check = None
            k = StepKind.ANALYSIS
        else:
            raise PlanningError(f"step {desc[:40]!r} has invalid kind {kind!r}")
        steps.append(PlanStep(
            id=uuid.uuid4().hex[:12],
            description=desc,
            kind=k,
            tool_name=tool_name,
            check=check,
        ))
    return Plan(task_id=task_id, steps=steps)


def _plan_messages(task: AgentTask, tools: list[ToolInfo], extra_context: str, last_error: str) -> list[dict]:
    content_parts = [
        f"Goal: {task.goal}",
        "",
        tool_catalog_text(tools),
    ]
    if extra_context:
        content_parts += ["", "Additional context:", extra_context]
    content_parts += [
        "",
        "Return the full plan as a single create_plan tool call.",
    ]
    if last_error:
        content_parts += ["", f"Previous attempt was rejected: {last_error}. Fix it."]
    return [
        {"role": "system", "content": (
            "You are a planner. You never execute anything - you only produce "
            "an executable plan via the create_plan tool."
        )},
        {"role": "user", "content": "\n".join(content_parts)},
    ]


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m is None:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None

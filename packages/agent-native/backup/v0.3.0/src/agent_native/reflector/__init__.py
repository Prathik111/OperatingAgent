"""Reflector - re-planning after a failed step (bounded).

Decision #1: replan cycles are bounded by max_replans (default 3). The bound
lives in Reflector (defense in depth) as a per-task counter; exceeding it
raises ReplanBudgetExhausted, which NativeAgent maps to a terminal FAILED
run with a visible reason - plus a REPLAN_BUDGET_EXHAUSTED event.

The new plan is produced from the old plan plus the failure reason, so the
model sees exactly what went wrong (not just the original goal).
"""

from __future__ import annotations

from ..llm import LLMClient
from ..planner import _extract_json, _plan_messages
from ..repository import TaskRepository
from ..types import AgentTask, Plan, ToolInfo, ToolSchema
from ..events import REPLANNING, AgentEvent


class ReplanBudgetExhausted(RuntimeError):
    pass


_REPLAN_TOOL_NAME = "create_plan"
_REPLAN_TOOL_DESCRIPTION = (
    "Create a REVISED step-by-step plan for the same goal, fixing the "
    "described failure. Rules: skip or adjust the failed step; do not repeat "
    "a step that already failed twice; keep kind='tool' steps using real tool "
    "names from the catalog with objective `check` descriptors, and "
    "kind='analysis' steps for pure reasoning."
)
_REPLAN_SCHEMA = {
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
                    "check": {"type": ["string", "null"]},
                },
                "required": ["description", "kind"],
            },
        }
    },
    "required": ["steps"],
}


def _replan_tool() -> ToolInfo:
    return ToolInfo(
        name=_REPLAN_TOOL_NAME,
        description=_REPLAN_TOOL_DESCRIPTION,
        schema=ToolSchema(input_schema=_REPLAN_SCHEMA, output_schema={}),
    )


class Reflector:
    def __init__(
        self,
        llm: LLMClient,
        repository: TaskRepository,
        max_replans: int = 3,
        on_event=None,
    ) -> None:
        self.llm = llm
        self.repository = repository
        self.max_replans = max_replans
        self.on_event = on_event
        self._counts: dict[str, int] = {}

    def replan_count(self, task_id: str) -> int:
        return self._counts.get(task_id, 0)

    async def replan(
        self,
        task: AgentTask,
        plan: Plan,
        failure_reason: str,
        tools: list[ToolInfo],
        *,
        extra_context: str = "",
    ) -> Plan:
        used = self._counts.get(task.id, 0)
        if used >= self.max_replans:
            raise ReplanBudgetExhausted(
                f"replan budget exhausted {used}/{self.max_replans}"
            )
        self._counts[task.id] = used + 1

        prior = "\n".join(
            f"- [{s.status.value}] {s.description}"
            + (f" (tool={s.tool_name})" if s.tool_name else "")
            for s in plan.steps
        )
        content_parts = [
            f"Goal: {task.goal}",
            "",
            f"Previous plan:\n{prior}",
            "",
            f"Failure to fix: {failure_reason}",
            "",
            "Return the revised plan as a single create_plan tool call.",
        ]
        if extra_context:
            content_parts.insert(0, f"Additional context: {extra_context}")

        messages = _plan_messages(task, tools, "", "")
        messages[-1] = {"role": "user", "content": "\n".join(content_parts)}

        response = await self.llm.complete(messages, tools=[_replan_tool()])
        new_plan = self._parse_replan(task.id, response, tools)
        if new_plan is None:
            raise ReplanBudgetExhausted("reflector produced an invalid revision")
        await self.repository.save_plan(new_plan)
        if self.on_event is not None:
            result = self.on_event(AgentEvent(
                kind=REPLANNING, task_id=task.id,
                payload={"attempt": used + 1, "reason": failure_reason},
            ))
            if result is not None:
                await result
        return new_plan

    def _parse_replan(self, task_id: str, response, tools: list[ToolInfo]) -> Plan | None:
        try:
            if response.wants_tool_call:
                call = response.tool_calls[0]  # type: ignore[index]
                if call.name != _REPLAN_TOOL_NAME:
                    return None
                data = call.arguments
            else:
                data = _extract_json(response.text or "")
                if data is None:
                    return None
            from ..planner import build_plan_from_data

            return build_plan_from_data(task_id, data, tools)
        except Exception:
            return None

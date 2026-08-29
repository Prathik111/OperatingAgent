"""ReactExecutor - the per-step Thought -> ToolCall -> Observation loop.

Bounded by max_calls_per_step (default 5). Every tool call passes the risk
gate first; REVIEW calls go through the ApprovalGateway (decision #4: timeout
auto-denies); BLOCKED calls abort the step for re-planning. Tool results are
appended as paired messages (see compactor for why), verification runs
against the step's check, and the context budget is checked before every LLM
turn.

All-or-nothing turns: if any call in a turn is blocked/denied the step aborts
before any assistant-tool_call message is appended, so the message history
can never contain a tool_call without its results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..approval import ApprovalGateway, emit_event
from ..compactor import ContextCompactor
from ..events import (
    COMPACTED,
    STEP_FAILED,
    STEP_SUCCEEDED,
    STEP_UNVERIFIABLE,
    TOOL_FINISHED,
    TOOL_STARTED,
    AgentEvent,
)
from ..llm import LLMClient
from ..mcp import MCPClient
from ..risk import RiskClassifier
from ..types import (
    AgentTask,
    ApprovalDecision,
    PlanStep,
    StepOutcomeStatus,
    StepStatus,
    ToolCallRequest,
    ToolCallResult,
)
from ..verifier import VerificationOutcome, Verifier


@dataclass(slots=True)
class StepOutcome:
    status: StepOutcomeStatus
    output: str | None = None
    tool_calls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    reason: str | None = None


class ReactExecutor:
    def __init__(
        self,
        llm: LLMClient,
        mcp: MCPClient,
        risk: RiskClassifier,
        verifier: Verifier,
        compactor: ContextCompactor,
        approval: ApprovalGateway | None = None,
        *,
        max_calls_per_step: int = 5,
        on_event=None,
    ) -> None:
        self.llm = llm
        self.mcp = mcp
        self.risk = risk
        self.verifier = verifier
        self.compactor = compactor
        self.approval = approval
        self.max_calls_per_step = max_calls_per_step
        self.on_event = on_event

    async def execute_step(
        self,
        task: AgentTask,
        step: PlanStep,
        tools: list,
        plan_context: str = "",
    ) -> StepOutcome:
        step.status = StepStatus.IN_PROGRESS
        messages = self._initial_messages(task, step, tools, plan_context)
        result: ToolCallResult | None = None
        last_tool_result: ToolCallResult | None = None
        tool_calls = 0
        llm_calls = 0
        tokens = 0

        for _turn in range(self.max_calls_per_step):
            if self.compactor.check_budget(messages):
                messages = self.compactor.compact(messages)
                await emit_event(self.on_event, AgentEvent(
                    kind=COMPACTED, task_id=task.id,
                    payload={"step_id": step.id},
                ))

            response = await self.llm.complete(messages, tools=tools)
            llm_calls += 1
            tokens += response.usage.total

            if not response.wants_tool_call:
                if response.text:
                    result = ToolCallResult(success=True, output=response.text)
                return await self._verify_and_finish(
                    task, step, last_tool_result or result, llm_calls, tool_calls, tokens
                )

            # All-or-nothing: abort before appending if any call is blocked/denied.
            gated: list[tuple] = []
            for call in response.tool_calls or []:
                request = ToolCallRequest(tool_name=call.name, arguments=call.arguments)
                level = self.risk.classify(task.id, request)
                if level.value == "blocked":
                    reason = f"tool {call.name!r} blocked by risk classifier"
                    await emit_event(self.on_event, AgentEvent(
                        kind=STEP_FAILED, task_id=task.id,
                        payload={"step_id": step.id, "reason": reason, "risk": level.value},
                    ))
                    return StepOutcome(StepOutcomeStatus.BLOCKED, reason=reason,
                                       llm_calls=llm_calls, tool_calls=tool_calls, tokens=tokens)
                if level.value == "review":
                    if self.approval is None:
                        reason = "approval required but no gateway"
                        await emit_event(self.on_event, AgentEvent(
                            kind=STEP_FAILED, task_id=task.id,
                            payload={"step_id": step.id, "reason": reason, "risk": level.value},
                        ))
                        return StepOutcome(StepOutcomeStatus.DENIED,
                                           reason=reason,
                                           llm_calls=llm_calls, tool_calls=tool_calls, tokens=tokens)
                    decision = await self.approval.request_approval(task.id, step)
                    if decision != ApprovalDecision.APPROVED:
                        reason = f"approval {decision.value}"
                        await emit_event(self.on_event, AgentEvent(
                            kind=STEP_FAILED, task_id=task.id,
                            payload={"step_id": step.id, "reason": reason, "risk": level.value},
                        ))
                        return StepOutcome(StepOutcomeStatus.DENIED, reason=reason,
                                           llm_calls=llm_calls, tool_calls=tool_calls, tokens=tokens)
                gated.append((call, request))

            executed: list[tuple] = []
            for call, request in gated:
                await emit_event(self.on_event, AgentEvent(
                    kind=TOOL_STARTED, task_id=task.id,
                    payload={"step_id": step.id, "tool": request.tool_name,
                             "arguments": request.arguments},
                ))
                result = await self.mcp.call_tool(request)
                last_tool_result = result
                tool_calls += 1
                await emit_event(self.on_event, AgentEvent(
                    kind=TOOL_FINISHED, task_id=task.id,
                    payload={"step_id": step.id, "tool": request.tool_name,
                             "success": result.success,
                             "output": _truncate(str(result.output), 1000)},
                ))
                executed.append((call, result))

            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name, "arguments": _dump_args(call.arguments)}}
                    for call, _r in executed
                ],
            }
            messages.append(assistant_msg)
            for call, call_result in executed:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(call_result.output),
                })

            if step.check and last_tool_result is not None \
                    and self.verifier.verify(step, last_tool_result) is VerificationOutcome.PASS:
                return await self._verify_and_finish(
                    task, step, last_tool_result, llm_calls, tool_calls, tokens
                )

        return StepOutcome(StepOutcomeStatus.MAX_CALLS_EXCEEDED,
                           reason=f"max_calls_per_step={self.max_calls_per_step} reached",
                           llm_calls=llm_calls, tool_calls=tool_calls, tokens=tokens)

    async def _verify_and_finish(
        self,
        task: AgentTask,
        step: PlanStep,
        result: ToolCallResult | None,
        llm_calls: int,
        tool_calls: int,
        tokens: int,
    ) -> StepOutcome:
        step.result = result
        if result is None:
            outcome = VerificationOutcome.UNVERIFIABLE if step.kind.value == "analysis" else VerificationOutcome.FAIL
        else:
            outcome = self.verifier.verify(step, result)
        output = str(result.output) if result else None

        if outcome == VerificationOutcome.PASS:
            step.status = StepStatus.DONE
            await emit_event(self.on_event, AgentEvent(
                kind=STEP_SUCCEEDED, task_id=task.id,
                payload={"step_id": step.id},
            ))
            return StepOutcome(StepOutcomeStatus.SUCCESS, output=output,
                               llm_calls=llm_calls, tool_calls=tool_calls, tokens=tokens)
        if outcome == VerificationOutcome.UNVERIFIABLE:
            step.status = StepStatus.DONE
            await emit_event(self.on_event, AgentEvent(
                kind=STEP_UNVERIFIABLE, task_id=task.id,
                payload={"step_id": step.id, "kind": step.kind.value},
            ))
            return StepOutcome(StepOutcomeStatus.SUCCESS, output=output,
                               llm_calls=llm_calls, tool_calls=tool_calls, tokens=tokens)
        step.status = StepStatus.FAILED
        reason = f"verification failed for check {step.check!r}"
        await emit_event(self.on_event, AgentEvent(
            kind=STEP_FAILED, task_id=task.id,
            payload={"step_id": step.id, "reason": reason},
        ))
        return StepOutcome(StepOutcomeStatus.VERIFY_FAIL, reason=reason,
                           llm_calls=llm_calls, tool_calls=tool_calls, tokens=tokens)

    def _initial_messages(
        self,
        task: AgentTask,
        step: PlanStep,
        tools: list,
        plan_context: str,
    ) -> list[dict]:
        from ..llm import SYSTEM_PROMPT
        from ..planner import tool_catalog_text

        content = f"Goal: {task.goal}\n\nExecute ONLY this step:\n{step.description}"
        if step.tool_name:
            content += f"\n(expected tool: {step.tool_name})"
        if step.check:
            content += f"\nVerification check: {step.check}"
        if plan_context:
            content += f"\n\nPlan context:\n{plan_context}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]


def _dump_args(arguments: dict) -> str:
    return json.dumps(arguments) if arguments else "{}"


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width] + "...[truncated]"

"""ApprovalGateway - user-in-the-loop gate for REVIEW-classified calls.

Decision #4: every approval request has a timeout (default 120s,
config.approval_timeout_s). On timeout the call is DENIED (auto-deny is the
safer default) and an APPROVAL_TIMED_OUT event is emitted so the UI/CLI can
show it - never a silent deny.

The gateway is transport-agnostic: it calls `on_event` (which a future
WebSocket layer can relay) and exposes `resolve()` for whoever owns the
decision side (CLI stdin, test harness, future ApprovalGateway/WS service).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..events import APPROVAL_REQUESTED, APPROVAL_RESOLVED, APPROVAL_TIMED_OUT, AgentEvent
from ..types import ApprovalDecision, PlanStep

EventCallback = Callable[[AgentEvent], Awaitable[None] | None]


@dataclass(slots=True)
class PendingApproval:
    task_id: str
    step_id: str
    description: str
    tool_name: str | None
    event: asyncio.Event = field(default_factory=asyncio.Event)
    decision: ApprovalDecision | None = None


async def emit_event(on_event: EventCallback | None, event: AgentEvent) -> None:
    if on_event is None:
        return
    result = on_event(event)
    if result is not None:
        await result


class ApprovalGateway:
    def __init__(
        self,
        timeout_s: float = 120.0,
        on_event: EventCallback | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.on_event = on_event
        self._pending: dict[str, PendingApproval] = {}

    @property
    def pending(self) -> dict[str, PendingApproval]:
        return self._pending

    async def request_approval(self, task_id: str, step: PlanStep) -> ApprovalDecision:
        approval = PendingApproval(
            task_id=task_id,
            step_id=step.id,
            description=step.description,
            tool_name=step.tool_name,
        )
        self._pending[step.id] = approval
        try:
            await emit_event(self.on_event, AgentEvent(
                kind=APPROVAL_REQUESTED, task_id=task_id,
                payload={"step_id": step.id, "description": step.description,
                         "tool_name": step.tool_name},
            ))
            decision = await self._wait_for_decision(approval)
            if decision == ApprovalDecision.TIMED_OUT:
                await emit_event(self.on_event, AgentEvent(
                    kind=APPROVAL_TIMED_OUT, task_id=task_id,
                    payload={"step_id": step.id, "timeout_s": self.timeout_s},
                ))
            else:
                await emit_event(self.on_event, AgentEvent(
                    kind=APPROVAL_RESOLVED, task_id=task_id,
                    payload={"step_id": step.id, "decision": decision.value},
                ))
            return decision
        finally:
            self._pending.pop(step.id, None)

    async def _wait_for_decision(self, approval: PendingApproval) -> ApprovalDecision:
        try:
            await asyncio.wait_for(approval.event.wait(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            return ApprovalDecision.TIMED_OUT
        if approval.decision is not None:
            return approval.decision
        return ApprovalDecision.TIMED_OUT

    def resolve(self, step_id: str, decision: ApprovalDecision) -> bool:
        """Called by the decision side; returns False if nothing was pending."""
        approval = self._pending.get(step_id)
        if approval is None:
            return False
        approval.decision = decision
        approval.event.set()
        return True

    async def close(self) -> None:
        for approval in self._pending.values():
            approval.decision = ApprovalDecision.TIMED_OUT
            approval.event.set()
        self._pending.clear()

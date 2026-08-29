"""Agent lifecycle events emitted by NativeAgent.

One dataclass with a `kind` discriminator (plus helpers) instead of a class
per event - the sink (CLI, future WebSocket, tests) switches on `kind`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

PLANNING_STARTED = "planning_started"
PLANNING_SUCCEEDED = "planning_succeeded"
PLANNING_FAILED = "planning_failed"
TOOL_STARTED = "tool_started"
TOOL_FINISHED = "tool_finished"
APPROVAL_REQUESTED = "approval_requested"
APPROVAL_RESOLVED = "approval_resolved"
APPROVAL_TIMED_OUT = "approval_timed_out"
STEP_SUCCEEDED = "step_succeeded"
STEP_FAILED = "step_failed"
STEP_UNVERIFIABLE = "step_unverifiable"
REPLANNING = "replanning"
REPLAN_BUDGET_EXHAUSTED = "replan_budget_exhausted"
COMPACTED = "compacted"
AGENT_FINISHED = "agent_finished"
RUN_FAILED = "run_failed"


@dataclass(slots=True)
class AgentEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
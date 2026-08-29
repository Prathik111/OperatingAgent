"""Local data types for agent-native.

Deliberately self-contained: nothing here is imported from packages/common.
These are this package's own request/result/task/enum shapes. Field names and
semantics are chosen for what this package actually needs, not for parity with
any other package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RiskLevel(str, Enum):
    """Output of RiskClassifier.classify()."""

    SAFE = "safe"
    REVIEW = "review"
    BLOCKED = "blocked"


class StepKind(str, Enum):
    """What a plan step is, which decides how (and whether) it can be verified.

    Decision #2: steps the planner cannot attach an objective check to are
    marked ANALYSIS (unverifiable). Verifier returns UNVERIFIABLE for those -
    it never silently passes them. TOOL steps carry a `check` descriptor.
    """

    TOOL = "tool"
    ANALYSIS = "analysis"


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


class StepOutcomeStatus(str, Enum):
    """Outcome of executing one plan step (ReactExecutor.execute_step)."""

    SUCCESS = "success"
    VERIFY_FAIL = "verify_fail"
    BLOCKED = "blocked"
    DENIED = "denied"
    MAX_CALLS_EXCEEDED = "max_calls_exceeded"


@dataclass(slots=True)
class ToolSchema:
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolInfo:
    name: str
    description: str
    schema: ToolSchema
    risk_level: str = "safe"


@dataclass(slots=True)
class ToolCallRequest:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolCallResult:
    success: bool
    output: Any
    error: str | None = None


@dataclass(slots=True)
class PlanStep:
    id: str
    description: str
    kind: StepKind
    tool_name: str | None = None
    check: str | None = None
    status: StepStatus = StepStatus.PENDING
    result: ToolCallResult | None = None


@dataclass(slots=True)
class Plan:
    task_id: str
    steps: list[PlanStep]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def step_map(self) -> dict[str, PlanStep]:
        return {s.id: s for s in self.steps}


@dataclass(slots=True)
class AgentTask:
    id: str
    goal: str
    thread_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:
    status: RunStatus
    output: str | None
    duration_ms: float
    llm_calls: int
    tool_calls: int
    total_tokens: int
    cost: float = 0.0
    replans: int = 0
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
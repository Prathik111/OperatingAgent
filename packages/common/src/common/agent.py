from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import AgentTrack, RunStatus


@dataclass(slots=True)
class AgentTask:
    id: str
    goal: str

    thread_id: str

    track: AgentTrack

    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)

    # Runtime-only execution controls.  They are deliberately not part of the
    # database task row: a task can have multiple attempts with different
    # modes while the task itself remains the user's durable turn.
    execution_mode: str = "new"  # new | continue | resume
    resume_value: Any = None
    resume_checkpoint_id: str | None = None
    resume_checkpoint_namespace: str | None = None
    completed_tool_calls: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:

    status: RunStatus

    output: str | None

    duration_ms: float

    llm_calls: int

    tool_calls: int

    total_tokens: int

    cost: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

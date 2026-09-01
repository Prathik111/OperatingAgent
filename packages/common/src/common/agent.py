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

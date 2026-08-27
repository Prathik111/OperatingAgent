"""Pydantic v2 request/response models for the HTTP surface.

These are the wire contracts only — the domain objects (``AgentTask``,
``ApprovalRequest``) stay as dataclasses in ``common``/the service layer, and
the routers translate between the two with the ``from_*`` constructors here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from common.agent import AgentTask
from common.enums import AgentTrack, RiskLevel


class CreateTaskRequest(BaseModel):
    """Body of ``POST /tasks``."""

    goal: str = Field(min_length=1, description="what the agent should accomplish")
    track: AgentTrack | None = Field(
        default=None, description="which orchestrator to use; server default if omitted"
    )
    thread_id: str | None = Field(
        default=None, description="conversation id to attach this task to; generated if omitted"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """A task plus its current run status (status lives on the run, not the task)."""

    id: str
    goal: str
    thread_id: str
    track: AgentTrack
    status: str | None = Field(default=None, description="latest run status, if a run exists")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @classmethod
    def from_task(cls, task: AgentTask, *, status: str | None) -> "TaskResponse":
        return cls(
            id=task.id,
            goal=task.goal,
            thread_id=task.thread_id,
            track=task.track,
            status=status,
            metadata=task.metadata,
            created_at=task.created_at,
        )


class ResolveApprovalRequest(BaseModel):
    """Body of ``POST /approvals/{id}/resolve``."""

    approved: bool
    note: str | None = None


class ApprovalResponse(BaseModel):
    id: str
    task_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: RiskLevel

    @classmethod
    def from_request(cls, request: Any) -> "ApprovalResponse":
        return cls(
            id=request.id,
            task_id=request.task_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            risk_level=request.risk_level,
        )


class HealthResponse(BaseModel):
    status: str
    repository: str
    tracks: list[str]

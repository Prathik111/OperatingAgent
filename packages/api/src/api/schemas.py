"""Pydantic v2 request/response models for the HTTP surface.

These are the wire contracts only — the domain objects (``AgentTask``,
``ApprovalRequest``) stay as dataclasses in ``common``/the service layer, and
the routers translate between the two with the ``from_*`` constructors here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from common.agent import AgentTask
from common.enums import AgentTrack, RiskLevel
from pydantic import BaseModel, ConfigDict, Field

from .repository.base import RunSummary, ThreadRecord


class CreateTaskRequest(BaseModel):
    """Body of ``POST /tasks``."""

    goal: str = Field(min_length=1, description="what the agent should accomplish")
    track: AgentTrack | None = Field(
        default=None,
        description=(
            "which orchestrator to use; server default if omitted. "
            "The native track invokes the same AgentService used by /native."
        ),
    )
    workspace: str | None = Field(
        default=None,
        description="existing directory the agent may access; server default if omitted",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Keep accepting the pre-thread-route body during client rollout. The field
    # is deliberately undocumented and is only read by the compatibility shim.
    model_config = ConfigDict(extra="allow")



class ResumeTaskRequest(BaseModel):
    """Body of ``POST /tasks/{task_id}/resume``."""

    resume_value: Any | None = Field(
        default=None,
        description="Value supplied to a pending LangGraph interrupt, if any",
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="Optional checkpoint id; latest checkpoint is used by default",
    )


class TaskResponse(BaseModel):
    """A task plus its current run status (status lives on the run, not the task)."""

    id: str
    goal: str
    thread_id: str
    workspace: str | None = None
    track: AgentTrack
    status: str | None = Field(default=None, description="latest run status, if a run exists")
    output: str | None = Field(default=None, description="latest run final answer")
    final_message: str | None = Field(
        default=None,
        description="latest run final assistant message",
    )
    error: str | None = Field(default=None, description="latest run error, if any")
    run_id: str | None = Field(default=None, description="latest run id")
    trace_id: str | None = Field(default=None, description="latest Langfuse trace id")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @classmethod
    def from_task(
        cls,
        task: AgentTask,
        *,
        status: str | None,
        run: RunSummary | None = None,
    ) -> TaskResponse:
        metadata = run.metadata if run is not None else {}
        return cls(
            id=task.id,
            goal=task.goal,
            thread_id=task.thread_id,
            workspace=str(
                task.metadata.get("workspace")
                or task.metadata.get("working_directory")
                or ""
            )
            or None,
            track=task.track,
            status=status,
            output=run.output if run is not None else None,
            final_message=run.output if run is not None else None,
            error=run.error if run is not None else None,
            run_id=run.run_id if run is not None else None,
            trace_id=(
                str(metadata.get("trace_id") or metadata.get("langfuse_trace_id"))
                if metadata.get("trace_id") or metadata.get("langfuse_trace_id")
                else None
            ),
            metadata=task.metadata,
            created_at=task.created_at,
        )


class ThreadResponse(BaseModel):
    """One conversation thread and its aggregate task count."""

    id: str
    title: str | None = None
    task_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, thread: ThreadRecord) -> ThreadResponse:
        return cls(
            id=thread.id,
            title=thread.title,
            task_count=thread.task_count,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
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
    def from_request(cls, request: Any) -> ApprovalResponse:
        return cls(
            id=request.id,
            task_id=request.task_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            risk_level=request.risk_level,
        )


class ThreadEventResponse(BaseModel):
    task_id: str
    type: str
    payload: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    repository: str
    tracks: list[str]

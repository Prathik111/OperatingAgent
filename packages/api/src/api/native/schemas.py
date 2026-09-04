"""Pydantic schemas for the native-track API.

Session-native shapes — mirrors agent-native's Session/RunResult/Event directly
so the frontend sees the same contract the CLI and the loop use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from observability import get_trace_url
from pydantic import AliasChoices, BaseModel, Field


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    agent: str = Field(default="build", description="AgentConfig name")
    title: str = Field(default="", description="Human title")
    workspace: str | None = Field(
        default=None,
        description="Existing folder file tools are confined to; server default if omitted",
        validation_alias=AliasChoices("workspace", "working_directory"),
    )


class SessionResponse(BaseModel):
    id: str
    agent: str
    title: str
    workspace: str

    @classmethod
    def from_native(cls, s: Any) -> SessionResponse:
        return cls(
            id=getattr(s, "id", ""),
            agent=getattr(s, "agent", "build"),
            title=getattr(s, "title", ""),
            workspace=getattr(s, "working_directory", ".") or ".",
        )


# ---------------------------------------------------------------------------
# Runs (defined before SessionWithRunsResponse for forward ref)
# ---------------------------------------------------------------------------
class RunResponse(BaseModel):
    run_id: str
    session_id: str = ""
    status: str
    turns: int = 0
    final_text: str = Field(
        default="",
        deprecated=True,
        description="Compatibility alias for final_message; use final_message",
    )
    final_message: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    model: str = ""
    retries: int = 0
    fallbacks: int = 0
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    trace_id: str = ""
    trace_url: str = ""

    @classmethod
    def from_native(cls, r: Any) -> RunResponse:
        usage = getattr(r, "usage", None)
        trace_id = getattr(r, "trace_id", "") or ""
        return cls(
            run_id=getattr(r, "run_id", ""),
            session_id=getattr(r, "session_id", "") or "",
            status=getattr(getattr(r, "status", ""), "value", str(getattr(r, "status", ""))),
            turns=int(getattr(r, "turns", 0) or 0),
            final_text=getattr(r, "final_text", "") or "",
            final_message=getattr(r, "final_text", "") or "",
            error=getattr(r, "error", "") or "",
            duration_seconds=float(getattr(r, "duration_seconds", 0.0) or 0.0),
            cost_usd=float(getattr(r, "cost_usd", 0.0) or 0.0),
            model=getattr(r, "model", "") or "",
            retries=int(getattr(r, "retries", 0) or 0),
            fallbacks=int(getattr(r, "fallbacks", 0) or 0),
            stop_reason=getattr(r, "stop_reason", "") or "",
            # ``RunResult`` exposes a nested Usage object; persisted native
            # ``RunRecord`` rows store the same values as top-level columns.
            input_tokens=int(
                getattr(usage, "input_tokens", getattr(r, "input_tokens", 0)) or 0
            ),
            output_tokens=int(
                getattr(usage, "output_tokens", getattr(r, "output_tokens", 0)) or 0
            ),
            cached_tokens=int(
                getattr(usage, "cached_tokens", getattr(r, "cached_tokens", 0)) or 0
            ),
            reasoning_tokens=int(
                getattr(usage, "reasoning_tokens", getattr(r, "reasoning_tokens", 0)) or 0
            ),
            trace_id=trace_id,
            trace_url=get_trace_url(trace_id) or "",
        )

    @classmethod
    def from_record(cls, rec: Any) -> RunResponse:
        # Postgres RunRecord shape
        trace_id = getattr(rec, "trace_id", "") or ""
        return cls(
            run_id=getattr(rec, "run_id", ""),
            session_id=getattr(rec, "session_id", ""),
            status=str(getattr(rec, "status", "")),
            turns=int(getattr(rec, "turns", 0) or 0),
            final_text=getattr(rec, "final_text", "") or "",
            final_message=getattr(rec, "final_text", "") or "",
            error=getattr(rec, "error", "") or "",
            duration_seconds=float(getattr(rec, "duration_seconds", 0.0) or 0.0),
            cost_usd=float(getattr(rec, "cost_usd", 0.0) or 0.0),
            model=getattr(rec, "model", "") or "",
            retries=int(getattr(rec, "retries", 0) or 0),
            fallbacks=0,
            stop_reason="",
            input_tokens=int(getattr(rec, "input_tokens", 0) or 0),
            output_tokens=int(getattr(rec, "output_tokens", 0) or 0),
            cached_tokens=int(getattr(rec, "cached_tokens", 0) or 0),
            reasoning_tokens=int(getattr(rec, "reasoning_tokens", 0) or 0),
            trace_id=trace_id,
            trace_url=get_trace_url(trace_id) or "",
        )


class SessionWithRunsResponse(SessionResponse):
    runs: list[RunResponse] = Field(default_factory=list)
    message_count: int = 0


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
class SendMessageRequest(BaseModel):
    message: str | None = Field(
        default=None,
        description="User text (legacy clients may send text)",
        validation_alias=AliasChoices("message", "text"),
    )
    media: list[dict[str, Any]] | None = Field(default=None, description="Optional Media parts [{data_base64,mime_type,detail}]")
    limits: LimitsRequest | None = None

    def resolved_text(self) -> str:
        return (self.message or "").strip()


class LimitsRequest(BaseModel):
    max_turns: int | None = None
    wall_clock_seconds: float | None = None
    max_cost_usd: float | None = None
    max_total_tokens: int | None = None
    max_retries: int | None = None
    max_parallel_tools: int | None = None
    helper_max_turns: int | None = None
    reasoning_effort: str | None = None
    plan_mode: bool | None = None


class ResumeRequest(BaseModel):
    limits: LimitsRequest | None = None


# (RunResponse already defined above)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
class EventResponse(BaseModel):
    sequence: int
    type: str
    session_id: str
    run_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    time: datetime | None = None


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------
class PermissionResponse(BaseModel):
    call_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    preview: str = ""
    reason: str = ""

    @classmethod
    def from_native(cls, req: Any) -> PermissionResponse:
        return cls(
            call_id=getattr(req, "call_id", ""),
            tool=getattr(req, "tool", ""),
            arguments=dict(getattr(req, "arguments", {}) or {}),
            preview=getattr(req, "preview", "") or "",
            reason=getattr(req, "reason", "") or "",
        )


class ResolvePermissionRequest(BaseModel):
    allowed: bool = Field(..., description="Whether the call may proceed")
    duration: str = Field(default="once", description="once|session|always")
    scope: str = Field(default="", description="Optional path scope to narrow a session/always grant")


# ---------------------------------------------------------------------------
# Messages / Conversation
# ---------------------------------------------------------------------------
class MessagePartResponse(BaseModel):
    part_type: str
    data: dict[str, Any]


class MessageResponse(BaseModel):
    id: str
    role: str
    parts: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Native health
# ---------------------------------------------------------------------------
class NativeHealthResponse(BaseModel):
    status: str
    database: str
    agents: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    langfuse_enabled: bool = False

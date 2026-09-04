from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .enums import RiskLevel


@dataclass(slots=True)
class ApprovalRequest:
    """A tool call waiting for a human decision."""

    id: str
    task_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: RiskLevel | None = None
    description: str | None = None
    run_id: str | None = None
    plan_step_id: str | None = None


@dataclass(slots=True, frozen=True)
class ApprovalRecord:
    """Durable approval state reconstructed from the application event log."""

    request: ApprovalRequest
    approved: bool | None = None
    note: str | None = None


class ApprovalHandler(Protocol):
    """Runtime approval boundary shared by orchestrators and the API."""

    async def request_approval(self, request: ApprovalRequest) -> bool: ...

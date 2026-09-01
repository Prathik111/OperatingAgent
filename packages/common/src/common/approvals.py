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


class ApprovalHandler(Protocol):
    """Runtime approval boundary shared by orchestrators and the API."""

    async def request_approval(self, request: ApprovalRequest) -> bool: ...

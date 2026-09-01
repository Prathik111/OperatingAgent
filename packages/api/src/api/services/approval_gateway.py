"""The human-in-the-loop approval gate.

``ApprovalGateway`` decides — using the deterministic ``RiskClassifier`` from
``common`` — whether a tool call can proceed automatically or must wait for a
human. A call below the configured threshold is auto-approved; a ``BLOCKED``
call is auto-denied; anything in between parks on an ``asyncio.Event`` until
someone calls :meth:`resolve_approval`.

Scope note: this gate is in-process and shared with the LangGraph executor, but
is not yet persisted. Pending approvals are therefore lost on process restart.
"""

from __future__ import annotations

import asyncio

from common.approvals import ApprovalRequest
from common.enums import RiskLevel
from common.risk import RiskClassifier
from common.tools import ToolCallRequest

from ..errors import ApprovalAlreadyResolved, ApprovalNotFound

#: Total order on risk, so "at or above the threshold" is a numeric comparison.
_ORDER = {RiskLevel.SAFE: 0, RiskLevel.REVIEW: 1, RiskLevel.BLOCKED: 2}


class _Pending:
    """Bookkeeping for a parked request: the waiter event and its outcome."""

    __slots__ = ("approved", "event", "note", "request", "resolved")

    def __init__(self, request: ApprovalRequest) -> None:
        self.request = request
        self.event = asyncio.Event()
        self.approved: bool = False
        self.note: str | None = None
        self.resolved: bool = False


class ApprovalGateway:
    def __init__(
        self,
        classifier: RiskClassifier | None = None,
        *,
        threshold: RiskLevel = RiskLevel.REVIEW,
    ) -> None:
        self._classifier = classifier or RiskClassifier()
        self._threshold = threshold
        self._pending: dict[str, _Pending] = {}
        self._lock = asyncio.Lock()

    async def request_approval(self, request: ApprovalRequest) -> bool:
        """Return whether the tool call may proceed, blocking for a human if needed.

        Auto-approves below the threshold, auto-denies ``BLOCKED``, otherwise
        registers the request and awaits :meth:`resolve_approval`.
        """
        level = request.risk_level or self._classifier.classify(
            ToolCallRequest(tool_name=request.tool_name, arguments=request.arguments)
        )
        request.risk_level = level

        if _ORDER[level] < _ORDER[self._threshold]:
            return True
        if level == RiskLevel.BLOCKED:
            return False

        pending = _Pending(request)
        async with self._lock:
            self._pending[request.id] = pending
        await pending.event.wait()
        return pending.approved

    async def resolve_approval(
        self, request_id: str, approved: bool, note: str | None = None
    ) -> None:
        """Resolve a parked request, waking whoever awaits it.

        Raises ``ApprovalNotFound`` for an unknown id and
        ``ApprovalAlreadyResolved`` if it was already decided.
        """
        async with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                raise ApprovalNotFound(request_id)
            if pending.resolved:
                raise ApprovalAlreadyResolved(request_id)
            pending.approved = approved
            pending.note = note
            pending.resolved = True
        pending.event.set()

    def list_pending(self) -> list[ApprovalRequest]:
        return [p.request for p in self._pending.values() if not p.resolved]

    def get(self, request_id: str) -> ApprovalRequest:
        pending = self._pending.get(request_id)
        if pending is None:
            raise ApprovalNotFound(request_id)
        return pending.request
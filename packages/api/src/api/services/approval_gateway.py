"""The human-in-the-loop approval gate.

``ApprovalGateway`` decides — using the deterministic ``RiskClassifier`` from
``common`` — whether a tool call can proceed automatically or must wait for a
human. A call below the configured threshold is auto-approved; a ``BLOCKED``
call is auto-denied; anything in between parks on an ``asyncio.Event`` until
someone calls :meth:`resolve_approval`.

Pending requests and decisions are mirrored into the repository event history,
then restored into in-process waiters when the API starts again.
"""

from __future__ import annotations

import asyncio
from typing import Any

from common.approvals import ApprovalRecord, ApprovalRequest
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
        repository: Any | None = None,
    ) -> None:
        self._classifier = classifier or RiskClassifier()
        self._threshold = threshold
        self._repository = repository
        self._pending: dict[str, _Pending] = {}
        self._lock = asyncio.Lock()

    async def restore(self) -> None:
        """Restore unresolved approvals after an API process restart."""
        if self._repository is None:
            return
        for request in await self._repository.list_pending_approvals():
            async with self._lock:
                self._pending.setdefault(request.id, _Pending(request))

    async def request_approval(self, request: ApprovalRequest) -> bool:
        """Return whether the tool call may proceed, blocking for a human if needed.

        Auto-approves below the threshold, auto-denies ``BLOCKED``, otherwise
        registers the request and awaits :meth:`resolve_approval`.
        """
        level = request.risk_level or self._classifier.classify(
            ToolCallRequest(tool_name=request.tool_name, arguments=request.arguments)
        )
        request.risk_level = level

        if self._repository is not None:
            record: ApprovalRecord | None = await self._repository.get_approval_state(
                request.id
            )
            if record is not None and record.approved is not None:
                return record.approved

        if _ORDER[level] < _ORDER[self._threshold]:
            return True
        if level == RiskLevel.BLOCKED:
            if self._repository is not None:
                await self._repository.save_approval_request(request)
                await self._repository.resolve_approval(request.id, False, "blocked by policy")
            return False

        async with self._lock:
            pending = self._pending.get(request.id)
            if pending is None:
                pending = _Pending(request)
                self._pending[request.id] = pending
                if self._repository is not None:
                    if request.run_id and request.plan_step_id:
                        await self._repository.save_approval(
                            request.run_id,
                            {
                                "id": request.id,
                                "plan_step_id": request.plan_step_id,
                                "reason": f"risk level {level.value}",
                            },
                        )
                    await self._repository.save_approval_request(request)
        await pending.event.wait()
        return pending.approved

    async def resolve_approval(
        self, request_id: str, approved: bool, note: str | None = None
    ) -> None:
        """Resolve a parked request, waking whoever awaits it.

        Raises ``ApprovalNotFound`` for an unknown id and
        ``ApprovalAlreadyResolved`` if it was already decided.
        Persists the decision before signaling so a failure leaves the request pending.
        """
        async with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                record = (
                    await self._repository.get_approval_state(request_id)
                    if self._repository is not None
                    else None
                )
                if record is None:
                    raise ApprovalNotFound(request_id)
                if record.approved is not None:
                    raise ApprovalAlreadyResolved(request_id)
                pending = _Pending(record.request)
                self._pending[request_id] = pending
            if pending.resolved:
                raise ApprovalAlreadyResolved(request_id)
            request = pending.request
        if self._repository is not None and request.run_id and request.plan_step_id:
            await self._repository.resolve_approval(
                {
                    "approval_id": request.id,
                    "approved": approved,
                    "note": note,
                }
            )
        if self._repository is not None:
            await self._repository.resolve_approval(request_id, approved, note)
        async with self._lock:
            # Re-validate under lock after persist
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

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
import logging
from typing import Any

from common.approvals import ApprovalRecord, ApprovalRequest
from common.enums import RiskLevel
from common.risk import RiskClassifier
from common.tools import ToolCallRequest

from ..errors import ApprovalAlreadyResolved, ApprovalNotFound

#: Total order on risk, so "at or above the threshold" is a numeric comparison.
_ORDER = {RiskLevel.SAFE: 0, RiskLevel.REVIEW: 1, RiskLevel.BLOCKED: 2}
log = logging.getLogger(__name__)


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
        requests = await self._repository.list_pending_approvals()
        for request in requests:
            async with self._lock:
                self._pending.setdefault(request.id, _Pending(request))
        if requests:
            log.info("restored %d pending approval request(s)", len(requests))

    async def request_approval(self, request: ApprovalRequest) -> bool:
        """Return whether the tool call may proceed, blocking for a human if needed.

        Auto-approves below the threshold, auto-denies ``BLOCKED``, otherwise
        registers the request and awaits :meth:`resolve_approval`.
        """
        level = request.risk_level or self._classifier.classify(
            ToolCallRequest(tool_name=request.tool_name, arguments=request.arguments)
        )
        request.risk_level = level
        log.debug(
            "approval evaluated request_id=%s task_id=%s tool=%s risk=%s threshold=%s",
            request.id,
            request.task_id,
            request.tool_name,
            level.value,
            self._threshold.value,
        )

        if self._repository is not None:
            record: ApprovalRecord | None = await self._repository.get_approval_state(
                request.id
            )
            if record is not None and record.approved is not None:
                log.info(
                    "approval request_id=%s already resolved durably approved=%s",
                    request.id,
                    record.approved,
                )
                return record.approved

        if _ORDER[level] < _ORDER[self._threshold]:
            log.debug("approval auto-approved request_id=%s below threshold", request.id)
            return True
        if level == RiskLevel.BLOCKED:
            log.warning(
                "approval auto-denied request_id=%s task_id=%s tool=%s by policy",
                request.id,
                request.task_id,
                request.tool_name,
            )
            if self._repository is not None:
                await self._repository.save_approval_request(request)
                await self._repository.resolve_approval(request.id, False, "blocked by policy")
            return False

        async with self._lock:
            pending = self._pending.get(request.id)
            if pending is None:
                if self._repository is not None:
                    try:
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
                    except Exception:
                        log.exception(
                            "approval request persistence failed request_id=%s; "
                            "request was not registered",
                            request.id,
                        )
                        raise
                pending = _Pending(request)
                self._pending[request.id] = pending
                log.info(
                    "approval requested request_id=%s task_id=%s tool=%s risk=%s",
                    request.id,
                    request.task_id,
                    request.tool_name,
                    level.value,
                )
            else:
                log.debug("approval request_id=%s joined existing waiter", request.id)
        log.debug("approval waiting request_id=%s", request.id)
        await pending.event.wait()
        log.info(
            "approval waiter released request_id=%s approved=%s",
            request.id,
            pending.approved,
        )
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
                    log.warning("approval resolution rejected unknown request_id=%s", request_id)
                    raise ApprovalNotFound(request_id)
                if record.approved is not None:
                    log.info(
                        "approval resolution rejected already-resolved request_id=%s",
                        request_id,
                    )
                    raise ApprovalAlreadyResolved(request_id)
                pending = _Pending(record.request)
                self._pending[request_id] = pending
            if pending.resolved:
                log.info(
                    "approval resolution rejected already-resolved request_id=%s",
                    request_id,
                )
                raise ApprovalAlreadyResolved(request_id)
            request = pending.request
        log.info(
            "persisting approval resolution request_id=%s task_id=%s approved=%s",
            request_id,
            request.task_id,
            approved,
        )
        try:
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
        except Exception:
            log.exception(
                "approval resolution persistence failed request_id=%s; request remains pending",
                request_id,
            )
            raise
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
        log.info(
            "approval resolved request_id=%s task_id=%s approved=%s",
            request_id,
            request.task_id,
            approved,
        )

    def list_pending(self) -> list[ApprovalRequest]:
        return [p.request for p in self._pending.values() if not p.resolved]

    def get(self, request_id: str) -> ApprovalRequest:
        pending = self._pending.get(request_id)
        if pending is None:
            raise ApprovalNotFound(request_id)
        return pending.request

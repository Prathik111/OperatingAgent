"""``ApprovalGateway`` — auto-approve / auto-deny / park-until-resolved."""

from __future__ import annotations

import asyncio

import pytest
from api.errors import ApprovalAlreadyResolved, ApprovalNotFound
from api.repository.memory import InMemoryTaskRepository
from api.services.approval_gateway import ApprovalGateway, ApprovalRequest
from common.enums import RiskLevel


def _req(request_id: str, tool_name: str, arguments: dict) -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id, task_id="t", tool_name=tool_name, arguments=arguments
    )


async def _wait_pending(gateway: ApprovalGateway) -> None:
    while not gateway.list_pending():
        await asyncio.sleep(0)


async def test_safe_call_auto_approved():
    gateway = ApprovalGateway()
    approved = await gateway.request_approval(
        _req("a1", "read_file", {"path": "/tmp/x"})
    )
    assert approved is True
    assert gateway.list_pending() == []


async def test_blocked_call_auto_denied():
    gateway = ApprovalGateway()
    # "sudo" trips the BLOCKED rule, which outranks the "shell" REVIEW rule.
    approved = await gateway.request_approval(
        _req("a2", "run_shell", {"command": "sudo reboot"})
    )
    assert approved is False
    assert gateway.list_pending() == []


async def test_review_call_parks_then_approves():
    gateway = ApprovalGateway()
    request = _req("a3", "delete_file", {"path": "/tmp/x"})
    waiter = asyncio.create_task(gateway.request_approval(request))
    await _wait_pending(gateway)

    assert [r.id for r in gateway.list_pending()] == ["a3"]
    assert gateway.get("a3").risk_level == RiskLevel.REVIEW

    await gateway.resolve_approval("a3", approved=True, note="looks fine")
    assert await waiter is True
    assert gateway.list_pending() == []


async def test_review_call_can_be_denied():
    gateway = ApprovalGateway()
    waiter = asyncio.create_task(
        gateway.request_approval(_req("a4", "delete_file", {"path": "/tmp/x"}))
    )
    await _wait_pending(gateway)

    await gateway.resolve_approval("a4", approved=False)
    assert await waiter is False


async def test_resolve_unknown_raises_not_found():
    gateway = ApprovalGateway()
    with pytest.raises(ApprovalNotFound):
        await gateway.resolve_approval("missing", approved=True)


async def test_double_resolve_raises_already_resolved():
    gateway = ApprovalGateway()
    waiter = asyncio.create_task(
        gateway.request_approval(_req("a5", "delete_file", {"path": "/tmp/x"}))
    )
    await _wait_pending(gateway)
    await gateway.resolve_approval("a5", approved=True)
    await waiter

    with pytest.raises(ApprovalAlreadyResolved):
        await gateway.resolve_approval("a5", approved=True)


async def test_threshold_can_be_lowered_to_safe():
    # With threshold SAFE, even a SAFE call must park for a human.
    gateway = ApprovalGateway(threshold=RiskLevel.SAFE)
    waiter = asyncio.create_task(
        gateway.request_approval(_req("a6", "read_file", {"path": "/tmp/x"}))
    )
    await _wait_pending(gateway)
    await gateway.resolve_approval("a6", approved=True)
    assert await waiter is True


async def test_approval_state_survives_gateway_reconstruction():
    repository = InMemoryTaskRepository()
    first = ApprovalGateway(repository=repository)
    request = _req("durable", "delete_file", {"path": "/tmp/x"})

    waiter = asyncio.create_task(first.request_approval(request))
    await _wait_pending(first)

    restored = ApprovalGateway(repository=repository)
    await restored.restore()
    assert [item.id for item in restored.list_pending()] == ["durable"]

    await restored.resolve_approval("durable", approved=True, note="approved")
    # A fresh graph attempt sees the durable decision without waiting on the
    # old process-local asyncio.Event.
    assert await restored.request_approval(request) is True

    # The original waiter remains valid in the process that created it too.
    await first.resolve_approval("durable", approved=True)
    assert await waiter is True

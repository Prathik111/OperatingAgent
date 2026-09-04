"""HTTP surface for the approval endpoints, driven through ``ASGITransport``."""

from __future__ import annotations

import asyncio

from api.services.approval_gateway import ApprovalRequest


async def _wait_pending(gateway) -> None:
    while not gateway.list_pending():
        await asyncio.sleep(0)


async def test_list_is_empty_by_default(client):
    resp = await client.get("/approvals")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_and_resolve_a_pending_request(client, approvals):
    request = ApprovalRequest(
        id="r1", task_id="t1", tool_name="delete_file", arguments={"path": "/x"}
    )
    waiter = asyncio.create_task(approvals.request_approval(request))
    await _wait_pending(approvals)

    listed = (await client.get("/approvals")).json()
    assert [r["id"] for r in listed] == ["r1"]
    assert listed[0]["risk_level"] == "review"

    resp = await client.post("/approvals/r1/resolve", json={"approved": True})
    assert resp.status_code == 200
    assert resp.json() == {"id": "r1", "approved": True}
    assert await waiter is True


async def test_get_unknown_approval_is_404(client):
    resp = await client.get("/approvals/missing")
    assert resp.status_code == 404


async def test_resolve_unknown_approval_is_404(client):
    resp = await client.post("/approvals/missing/resolve", json={"approved": True})
    assert resp.status_code == 404


async def test_double_resolve_is_409(client, approvals):
    request = ApprovalRequest(
        id="r2", task_id="t1", tool_name="delete_file", arguments={"path": "/x"}
    )
    waiter = asyncio.create_task(approvals.request_approval(request))
    await _wait_pending(approvals)

    first = await client.post("/approvals/r2/resolve", json={"approved": True})
    assert first.status_code == 200
    await waiter

    second = await client.post("/approvals/r2/resolve", json={"approved": True})
    assert second.status_code == 409

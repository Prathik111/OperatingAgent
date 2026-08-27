"""Approval endpoints — list, read, and resolve pending human gates.

These drive the in-process ``ApprovalGateway``. Because the gate is not yet
wired into the orchestrators (see the package README), in normal operation the
pending list is empty; the endpoints exist so the surface is complete and the
gate is exercisable end to end.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_approval_gateway
from ..schemas import ApprovalResponse, ResolveApprovalRequest
from ..services.approval_gateway import ApprovalGateway

router = APIRouter(prefix="/approvals", tags=["approvals"])


@router.get("", response_model=list[ApprovalResponse])
async def list_approvals(
    gateway: ApprovalGateway = Depends(get_approval_gateway),
) -> list[ApprovalResponse]:
    return [ApprovalResponse.from_request(r) for r in gateway.list_pending()]


@router.get("/{request_id}", response_model=ApprovalResponse)
async def get_approval(
    request_id: str,
    gateway: ApprovalGateway = Depends(get_approval_gateway),
) -> ApprovalResponse:
    return ApprovalResponse.from_request(gateway.get(request_id))  # 404 if unknown


@router.post("/{request_id}/resolve")
async def resolve_approval(
    request_id: str,
    body: ResolveApprovalRequest,
    gateway: ApprovalGateway = Depends(get_approval_gateway),
) -> dict:
    # Raises ApprovalNotFound (404) / ApprovalAlreadyResolved (409).
    await gateway.resolve_approval(request_id, body.approved, body.note)
    return {"id": request_id, "approved": body.approved}

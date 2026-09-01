"""Approval endpoints — list, read, and resolve pending human gates.

These drive the same in-process ``ApprovalGateway`` awaited by the LangGraph
executor. Resolving a pending request therefore lets the graph continue.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..dependencies import get_approval_gateway
from ..schemas import ApprovalResponse, ResolveApprovalRequest
from ..services.approval_gateway import ApprovalGateway

router = APIRouter(prefix="/approvals", tags=["approvals"])

ApprovalGatewayDep = Annotated[ApprovalGateway, Depends(get_approval_gateway)]


@router.get("", response_model=list[ApprovalResponse])
async def list_approvals(
    gateway: ApprovalGatewayDep,
) -> list[ApprovalResponse]:
    return [ApprovalResponse.from_request(r) for r in gateway.list_pending()]


@router.get("/{request_id}", response_model=ApprovalResponse)
async def get_approval(
    request_id: str,
    gateway: ApprovalGatewayDep,
) -> ApprovalResponse:
    return ApprovalResponse.from_request(gateway.get(request_id))  # 404 if unknown


@router.post("/{request_id}/resolve")
async def resolve_approval(
    request_id: str,
    body: ResolveApprovalRequest,
    gateway: ApprovalGatewayDep,
) -> dict:
    # Raises ApprovalNotFound (404) / ApprovalAlreadyResolved (409).
    await gateway.resolve_approval(request_id, body.approved, body.note)
    return {"id": request_id, "approved": body.approved}

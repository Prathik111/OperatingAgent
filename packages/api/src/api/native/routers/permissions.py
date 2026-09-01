"""Native permission endpoints — human-in-the-loop for mutating tools."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..dependencies import get_native_service
from ..schemas import PermissionResponse, ResolvePermissionRequest

router = APIRouter(prefix="/native/permissions", tags=["native-permissions"])

NativeServiceDep = Annotated[object, Depends(get_native_service)]


@router.get("", response_model=list[PermissionResponse])
async def list_permissions(
    service: NativeServiceDep,
    session_id: str = "",
) -> list[PermissionResponse]:
    # Pending is global in PermissionManager; filter by session if given
    pending = service.pending_permissions()
    if session_id:
        # PermissionRequest doesn't carry session_id, so we filter via
        # events that requested it — approximate by returning all when filtered
        # and let frontend filter by its known session. Keep contract simple.
        pass
    return [PermissionResponse.from_native(r) for r in pending]


@router.get("/{call_id}", response_model=PermissionResponse)
async def get_permission(call_id: str, service: NativeServiceDep) -> PermissionResponse:
    from fastapi import HTTPException

    for req in service.pending_permissions():
        if getattr(req, "call_id", "") == call_id:
            return PermissionResponse.from_native(req)
    raise HTTPException(status_code=404, detail=f"permission '{call_id}' not found or not pending")


@router.post("/{call_id}", response_model=dict)
async def resolve_permission(
    call_id: str,
    body: ResolvePermissionRequest,
    service: NativeServiceDep,
) -> dict:
    from fastapi import HTTPException

    # Validate existence first for 404
    pending_ids = {getattr(r, "call_id", "") for r in service.pending_permissions()}
    if call_id not in pending_ids:
        raise HTTPException(status_code=404, detail=f"permission '{call_id}' not found or not pending")

    duration_map = {
        "once": "once",
        "session": "session",
        "always": "always",
    }
    dur_str = duration_map.get(body.duration.lower().strip(), "once")

    # Map to PermissionDuration enum defensively
    try:
        from agent_native.permissions import PermissionDuration

        dur = PermissionDuration(dur_str)
    except Exception:
        from agent_native.permissions import PermissionDuration

        dur = PermissionDuration.ONCE

    await service.resolve_permission(call_id, bool(body.allowed), duration=dur, scope=body.scope or "")
    return {"call_id": call_id, "allowed": bool(body.allowed), "duration": dur_str, "scope": body.scope or ""}

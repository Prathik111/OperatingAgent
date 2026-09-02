"""Native run endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ..dependencies import get_native_service
from ..schemas import RunResponse

router = APIRouter(prefix="/native", tags=["native-runs"])

NativeServiceDep = Annotated[Any, Depends(get_native_service)]


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, service: NativeServiceDep) -> RunResponse:
    from fastapi import HTTPException

    db = service.runtime.database
    rec = await db.get_run(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    # RunRecord vs RunResult — normalize via from_record
    try:
        return RunResponse.from_native(rec)
    except Exception:
        return RunResponse.from_record(rec)


@router.get("/sessions/{session_id}/runs", response_model=list[RunResponse])
async def list_runs(session_id: str, service: NativeServiceDep) -> list[RunResponse]:
    from fastapi import HTTPException

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
    recs = await db.list_runs(session_id=session_id, limit=0)
    out: list[RunResponse] = []
    for r in recs:
        try:
            out.append(RunResponse.from_native(r))
        except Exception:
            out.append(RunResponse.from_record(r))
    return out

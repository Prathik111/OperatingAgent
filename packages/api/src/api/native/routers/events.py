"""Native event replay + live stream.

Two modes:
- GET /native/sessions/{id}/events?from=0        -> JSON replay (no stream)
- GET /native/sessions/{id}/events?from=0&stream=1 -> SSE catch-up then live
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..dependencies import get_native_service
from ..schemas import EventResponse

router = APIRouter(prefix="/native/sessions", tags=["native-events"])

NativeServiceDep = Annotated[Any, Depends(get_native_service)]


def _event_to_response(e: object) -> EventResponse:
    return EventResponse(
        sequence=int(getattr(e, "sequence", 0) or 0),
        type=str(getattr(e, "type", "")),
        session_id=str(getattr(e, "session_id", "")),
        run_id=str(getattr(e, "run_id", "") or ""),
        data=dict(getattr(e, "data", {}) or {}),
        time=getattr(e, "time", None),
    )


def _event_to_sse(e: object) -> dict:
    import json

    ev = _event_to_response(e)
    # Use pydantic's JSON-safe dump for data/time
    payload = ev.model_dump(mode="json")
    return {
        "id": str(ev.sequence),
        "event": ev.type,
        "data": json.dumps(payload, ensure_ascii=False, default=str),
    }


@router.get("/{session_id}/events")
async def get_events(
    session_id: str,
    service: NativeServiceDep,
    from_seq: int = Query(default=0, alias="from", ge=0, description="Last sequence the client has"),
    stream: int = Query(default=0, ge=0, le=1, description="1 to keep SSE open after catch-up"),
):
    from fastapi import HTTPException

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")

    if not stream:
        events = await db.load_events(session_id, after_sequence=from_seq)
        return JSONResponse(content=[_event_to_response(e).model_dump(mode="json") for e in events])

    # SSE: replay then live subscribe
    async def event_source():
        # Replay stored tail first
        for e in await db.load_events(session_id, after_sequence=from_seq):
            yield _event_to_sse(e)
        # Then subscribe live; de-duplicate by sequence
        last = max([from_seq] + [int(getattr(e, "sequence", 0) or 0) for e in await db.load_events(session_id, after_sequence=from_seq)], default=from_seq)
        async for e in service.subscribe(session_id, from_sequence=last):
            yield _event_to_sse(e)

    return EventSourceResponse(
        event_source(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        ping=15,
    )

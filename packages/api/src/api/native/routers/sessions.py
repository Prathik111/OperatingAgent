"""Native session endpoints — the transcript is the state."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status

from pydantic import BaseModel

from ..dependencies import get_native_service
from ..schemas import CreateSessionRequest, SessionResponse, SessionWithRunsResponse, RunResponse

router = APIRouter(prefix="/native/sessions", tags=["native-sessions"])

NativeServiceDep = Annotated[object, Depends(get_native_service)]
Limit = Annotated[int, Query(ge=0, le=500)]
Offset = Annotated[int, Query(ge=0)]


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionRequest, service: NativeServiceDep) -> SessionResponse:
    session = await service.create_session(
        agent=body.agent or "build",
        title=body.title or "",
        working_directory=body.working_directory or ".",
    )
    return SessionResponse.from_native(session)


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    request: Request,
    service: NativeServiceDep,
    working_directory: str = Query(default="", description="Filter by working_directory (exact)"),
    limit: Limit = 100,
    offset: Offset = 0,
) -> list[SessionResponse]:
    db = service.runtime.database
    sessions = await db.list_sessions(working_directory=working_directory, limit=0)
    # offset/limit in Python so working_directory filter is consistent across backends
    sliced = sessions[offset : offset + limit] if limit else sessions[offset:]
    return [SessionResponse.from_native(s) for s in sliced]


@router.get("/{session_id}", response_model=SessionWithRunsResponse)
async def get_session(session_id: str, service: NativeServiceDep) -> SessionWithRunsResponse:
    from fastapi import HTTPException

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
    conversation = await db.load_conversation(session_id)
    runs = await db.list_runs(session_id=session_id, limit=20)
    return SessionWithRunsResponse(
        **SessionResponse.from_native(session).model_dump(),
        runs=[RunResponse.from_record(r) for r in runs],
        message_count=len(conversation.messages),
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, service: NativeServiceDep):
    from fastapi import HTTPException, Response

    deleted = await service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class ForkSessionRequest(BaseModel):
    title: str = ""


@router.post("/{session_id}/fork", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def fork_session(
    session_id: str,
    service: NativeServiceDep,
    body: ForkSessionRequest | None = None,
) -> SessionResponse:
    from fastapi import HTTPException

    title = body.title if body else ""
    try:
        forked = await service.fork_session(session_id, title=title)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    return SessionResponse.from_native(forked)


@router.get("/{session_id}/conversation")
async def get_conversation(session_id: str, service: NativeServiceDep):
    from fastapi import HTTPException

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
    conv = await service.get_conversation(session_id)
    # Serialize messages lightly for the frontend
    out = []
    for m in conv.messages:
        parts = []
        for p in m.parts:
            kind = getattr(getattr(p, "part_type", None), "value", str(type(p).__name__))
            # Normalize to dict for JSON
            if hasattr(p, "text"):
                parts.append({"part_type": kind, "text": getattr(p, "text", ""), "hidden": getattr(p, "hidden", False)})
            elif hasattr(p, "name"):
                parts.append({
                    "part_type": kind,
                    "id": getattr(p, "id", ""),
                    "name": getattr(p, "name", ""),
                    "arguments": getattr(p, "arguments", {}),
                    "status": getattr(getattr(p, "status", ""), "value", str(getattr(p, "status", ""))),
                    "output": getattr(p, "output", ""),
                    "error": getattr(p, "error", ""),
                })
            elif hasattr(p, "summary"):
                parts.append({"part_type": kind, "summary": getattr(p, "summary", "")})
            elif hasattr(p, "data"):
                parts.append({"part_type": kind, "mime_type": getattr(p, "mime_type", ""), "detail": getattr(p, "detail", ""), "data": getattr(p, "data", "")[:200] + ("..." if len(getattr(p, "data", "")) > 200 else "")})
            else:
                parts.append({"part_type": kind, "data": str(p)})
        out.append({
            "id": getattr(m, "id", ""),
            "role": getattr(getattr(m, "role", ""), "value", str(getattr(m, "role", ""))),
            "parts": parts,
            "model": getattr(m, "model", "") or "",
            "created_at": getattr(m, "created_at", None).isoformat() if getattr(m, "created_at", None) else None,
        })
    return {"session_id": session_id, "messages": out}

"""Native messages — send a prompt and stream the run.

POST /native/sessions/{id}/messages  -> SSE of the run's events, ending on RUN_FINISHED.
POST /native/sessions/{id}/resume    -> resume + optional SSE via /events.
POST /native/sessions/{id}/cancel    -> cancel an in-flight run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..dependencies import get_native_service
from ..schemas import ResumeRequest, RunResponse, SendMessageRequest
from ..runtime import attach_mcp_tools

router = APIRouter(prefix="/native/sessions", tags=["native-messages"])

NativeServiceDep = Annotated[Any, Depends(get_native_service)]


def _limits_from_request(limits: object | None):
    if limits is None:
        return None
    try:
        from agent_native.loop import Limits
    except Exception:
        return None

    kwargs: dict = {}
    for field in ("max_turns", "wall_clock_seconds", "max_cost_usd", "max_total_tokens", "max_retries", "max_parallel_tools", "helper_max_turns", "reasoning_effort", "plan_mode"):
        val = getattr(limits, field, None)
        if val is not None:
            kwargs[field] = val
    if not kwargs:
        return None
    try:
        return Limits(**kwargs)
    except Exception:
        return None


def _media_from_request(media: list[dict] | None):
    if not media:
        return None
    try:
        from agent_native.conversation import media_part

        out = []
        for item in media:
            data = item.get("data") or item.get("data_base64") or ""
            mime = item.get("mime_type") or item.get("mimeType") or "image/png"
            detail = item.get("detail") or ""
            if not data:
                continue
            # data may already be base64 string; media_part handles both bytes and str
            out.append(media_part(data, mime_type=mime, detail=detail))
        return out or None
    except Exception:
        return None


def _event_to_sse_dict(e: object) -> dict:
    seq = int(getattr(e, "sequence", 0) or 0)
    typ = str(getattr(e, "type", ""))
    # Build minimal JSON payload matching EventResponse
    payload = {
        "sequence": seq,
        "type": typ,
        "session_id": str(getattr(e, "session_id", "")),
        "run_id": str(getattr(e, "run_id", "") or ""),
        "data": dict(getattr(e, "data", {}) or {}),
        "time": getattr(e, "time", None).isoformat() if getattr(e, "time", None) else None,
    }
    return {"id": str(seq), "event": typ, "data": json.dumps(payload, ensure_ascii=False, default=str)}


@router.post("/{session_id}/messages")
async def send_message(
    session_id: str,
    body: SendMessageRequest,
    service: NativeServiceDep,
    request: Request,
):
    from fastapi import HTTPException

    text = body.resolved_text()
    if not text and not body.media:
        raise HTTPException(status_code=422, detail="message or text is required")

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")

    # Attach MCP tools once per working_directory (lazy, in-process, no network)
    try:
        await attach_mcp_tools(service.runtime, working_directory=getattr(session, "working_directory", "."))
    except Exception:
        pass

    # Build limits and optional media-aware message.
    # AgentService.send_message currently takes text only; media is handled by
    # building a user_message with Media parts before invoking the loop.
    # For backwards compat, we inject media via a direct DB save + emit path
    # when media is present, then call send_message with the text. The loop's
    # conversation will already contain the media message.
    media_parts = _media_from_request(body.media)
    limits = _limits_from_request(body.limits)

    # Prepare a cancellation that the cancel endpoint can flip (stored on app.state)
    from agent_native.loop import Cancellation

    cancellation = Cancellation()
    # Register cancellation on the service runtime so POST /cancel can find it
    # Keyed by session_id -> Cancellation
    cancels: dict = getattr(request.app.state, "native_cancels", None)
    if cancels is None:
        request.app.state.native_cancels = cancels = {}
    cancels[session_id] = cancellation

    # We need to capture the run_id that send_message mints. Wrap service.send_message
    # to emit with that run_id, then stream events for that run_id.

    # Background run so we can stream events concurrently
    async def run_in_background():
        try:
            if media_parts:
                # Persist a media-bearing user message before the text turn so the
                # loop sees images/documents on its next render. We reuse the
                # conversation's user_message factory.
                from agent_native.conversation import user_message
                from agent_native.events import EventType

                media_msg = user_message(session_id, text=text, media=media_parts)
                await db.save_message(media_msg)
                # Also emit a MESSAGE_ADDED for observability (run_id will be minted inside send_message,
                # so this is a session-scoped marker, not run-scoped)
                try:
                    await service.runtime.events.emit(session_id, EventType.MESSAGE_ADDED, {"id": media_msg.id, "role": "user", "has_media": True}, run_id="")
                except Exception:
                    pass
                # Now send a lightweight follow-up text so the loop has a user turn to answer
                # The media message already carries the user text; send text as-is
                # (two user turns render correctly, and avoids double-counting complexity).
                result = await service.send_message(session_id, text, limits=limits, cancellation=cancellation)
            else:
                result = await service.send_message(session_id, text, limits=limits, cancellation=cancellation)
            return result
        finally:
            # Unregister cancellation when run ends
            cancels.pop(session_id, None)

    background = asyncio.create_task(run_in_background())
    # Ensure background is awaited even if client disconnects
    background_set: set = getattr(request.app.state, "native_background", None)
    if background_set is None:
        request.app.state.native_background = background_set = set()
    background_set.add(background)
    background.add_done_callback(background_set.discard)

    # Stream events for this session until the top-level RUN_FINISHED for this run.
    # We don't know run_id upfront (minted inside send_message), so we stream the
    # session's events and stop only when we see RUN_FINISHED for a non-helper run
    # that was emitted after we started. We also watch background completion to emit
    # a final sentinel if the run ended without a RUN_FINISHED (error path).

    # Snapshot the current tail so we only stream this run's events
    try:
        tip = max([0] + [int(getattr(e, "sequence", 0) or 0) for e in await db.load_events(session_id, after_sequence=0)], default=0)
    except Exception:
        tip = 0

    async def event_source():
        # Replay nothing — we start from tip; background will emit from tip+1
        async for event in service.subscribe(session_id, from_sequence=tip):
            yield _event_to_sse_dict(event)
            # Stop on top-level RUN_FINISHED (helper runs contain "/")
            typ = str(getattr(event, "type", ""))
            run_id = str(getattr(event, "run_id", "") or "")
            if typ == "run_finished" and "/" not in run_id:
                break
            # Also stop if background done and we've drained events up to its finish
            if background.done() and typ in ("run_finished", "error"):
                # Give a moment for final events to flush
                await asyncio.sleep(0.05)
                # If no new events arrive quickly, the loop will idle; break on background done
                # We don't break immediately to avoid cutting off the receipt event
                if background.done():
                    # Check if next event would be the terminal; if already yielded it, break
                    if typ == "run_finished":
                        break
        # Ensure background is awaited (propagate exception if any, but don't crash stream)
        try:
            await background
        except Exception:
            pass

    return EventSourceResponse(
        event_source(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        ping=15,
    )


@router.post("/{session_id}/resume")
async def resume_run(
    session_id: str,
    service: NativeServiceDep,
    request: Request,
    body: ResumeRequest | None = None,
):
    from fastapi import HTTPException

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")

    limits = _limits_from_request(body.limits if body else None)

    from agent_native.loop import Cancellation

    cancellation = Cancellation()
    cancels: dict = getattr(request.app.state, "native_cancels", None)
    if cancels is None:
        request.app.state.native_cancels = cancels = {}
    cancels[session_id] = cancellation
    try:
        result = await service.resume_run(session_id, limits=limits, cancellation=cancellation)
    finally:
        cancels.pop(session_id, None)

    return JSONResponse(content=RunResponse.from_native(result).model_dump(mode="json"))


@router.post("/{session_id}/cancel", status_code=202)
async def cancel_run(session_id: str, request: Request, service: NativeServiceDep):
    from fastapi import HTTPException

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")

    cancels: dict = getattr(request.app.state, "native_cancels", {}) or {}
    cancellation = cancels.get(session_id)
    if cancellation is None:
        # No in-flight run to cancel — idempotent 202 with hint
        return JSONResponse(status_code=202, content={"session_id": session_id, "cancelled": False, "reason": "no active run"})
    try:
        cancellation.cancel()
    except Exception:
        pass
    return JSONResponse(status_code=202, content={"session_id": session_id, "cancelled": True})

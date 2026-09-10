"""Native messages — send a prompt and stream the run.

POST /native/sessions/{id}/messages  -> SSE of the run's events, ending on RUN_FINISHED.
POST /native/sessions/{id}/resume    -> resume + optional SSE via /events.
POST /native/sessions/{id}/cancel    -> cancel an in-flight run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..dependencies import get_native_service
from ..runtime import attach_mcp_tools
from ..schemas import ResumeRequest, RunResponse, SendMessageRequest

router = APIRouter(prefix="/native/sessions", tags=["native-messages"])

NativeServiceDep = Annotated[Any, Depends(get_native_service)]
log = logging.getLogger(__name__)


def _cancel_registry(request: Request) -> dict[str, Any]:
    """In-flight cancellations keyed by run id — never by session.

    Two runs in one session each get their own ``Cancellation`` here, so
    cancelling one can never flip the other. ``native_cancels`` keeps its
    historical name; its keys changed from session ids to run ids.
    """
    cancels = getattr(request.app.state, "native_cancels", None)
    if not isinstance(cancels, dict):
        cancels = {}
        request.app.state.native_cancels = cancels
    return cancels


def _cancel_index(request: Request) -> dict[str, set[str]]:
    """Session id -> in-flight run ids, for session-wide cancellation."""
    index = getattr(request.app.state, "native_cancel_sessions", None)
    if not isinstance(index, dict):
        index = {}
        request.app.state.native_cancel_sessions = index
    return index


def _register_cancel(request: Request, session_id: str, run_id: str, cancellation: Any) -> None:
    _cancel_registry(request)[run_id] = cancellation
    _cancel_index(request).setdefault(session_id, set()).add(run_id)


def _unregister_cancel(request: Request, session_id: str, run_id: str) -> None:
    _cancel_registry(request).pop(run_id, None)
    run_ids = _cancel_index(request).get(session_id)
    if run_ids is not None:
        run_ids.discard(run_id)
        if not run_ids:
            _cancel_index(request).pop(session_id, None)


def _resume_lock(request: Request, session_id: str):
    """Per-session lock serializing concurrent resumes of one session.

    Two resumes racing would both continue the same interrupted run id and
    interleave two loops on one transcript. The lock makes the second resume
    observe the first one's RUN_FINISHED and return its receipt instead.
    """
    import asyncio

    locks = getattr(request.app.state, "native_resume_locks", None)
    if not isinstance(locks, dict):
        locks = {}
        request.app.state.native_resume_locks = locks
    lock = locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[session_id] = lock
    return lock


def _limits_from_request(limits: object | None):
    if limits is None:
        return None
    try:
        from agent_native.loop import Limits
    except ImportError:
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
    except (TypeError, ValueError):
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
    except (ImportError, TypeError, ValueError):
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
        "time": (
            event_time.isoformat() if (event_time := getattr(e, "time", None)) else None
        ),
    }
    return {"id": str(seq), "event": typ, "data": json.dumps(payload, ensure_ascii=False, default=str)}


def _run_receipt_to_sse_dict(result: object) -> dict:
    payload = RunResponse.from_native(result).model_dump(mode="json")
    return {
        "id": str(getattr(result, "run_id", "") or "receipt"),
        "event": "run_receipt",
        "data": json.dumps(payload, ensure_ascii=False, default=str),
    }


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
    except Exception as exc:  # noqa: BLE001 - MCP attachment is optional
        log.debug("MCP attachment skipped: %s", exc)

    # Build limits and optional media-aware message. AgentService persists one
    # user turn containing both text and media, avoiding duplicate user turns.
    media_parts = _media_from_request(body.media)
    limits = _limits_from_request(body.limits)

    # Snapshot the event cursor BEFORE the run starts. The bus subscription
    # below registers before replaying, so with the cursor predating the run,
    # no event the run emits can be missed and none is duplicated. Capturing
    # the cursor after starting the run (as before) drops whatever the run
    # emitted in between.
    try:
        tip = max([0] + [int(getattr(e, "sequence", 0) or 0) for e in await db.load_events(session_id, after_sequence=0)], default=0)
    except Exception as exc:  # noqa: BLE001 - event history is best effort
        log.debug("Could not determine native event tail: %s", exc)
        tip = 0

    # Mint the run id up front and pin it through the whole request, so the
    # cancellation registry and the event stream both name exactly this run.
    import uuid

    run_id = "run_" + uuid.uuid4().hex[:8]

    # Prepare a cancellation that the cancel endpoint can flip, keyed by run
    # id so concurrent runs in one session stay independently controllable.
    from agent_native.loop import Cancellation

    cancellation = Cancellation()
    _register_cancel(request, session_id, run_id, cancellation)

    # Background run so we can stream events concurrently
    run_result: Any | None = None

    async def run_in_background():
        nonlocal run_result
        try:
            run_result = await service.send_message(
                session_id,
                text,
                limits=limits,
                cancellation=cancellation,
                media=media_parts,
                run_id=run_id,
            )
            try:
                lf = getattr(service.runtime.monitoring, "langfuse_client", None)
                if lf is not None:
                    lf.flush()
            except Exception as exc:  # noqa: BLE001 - flushing is best effort
                log.debug("native Langfuse flush failed: %s", exc)
            return run_result
        finally:
            # Unregister only this run's cancellation; other runs are untouched.
            _unregister_cancel(request, session_id, run_id)

    background = asyncio.create_task(run_in_background())
    # Ensure background is awaited even if client disconnects
    existing_background = getattr(request.app.state, "native_background", None)
    background_set: set[asyncio.Task[Any]] = (
        existing_background if isinstance(existing_background, set) else set()
    )
    request.app.state.native_background = background_set
    background_set.add(background)
    background.add_done_callback(background_set.discard)

    def _belongs_to_run(event_run_id: str) -> bool:
        # Our run plus the helpers it spawns ("<run_id>/..."); other
        # concurrent runs in the session must never end or pollute this stream.
        return event_run_id == run_id or event_run_id.startswith(run_id + "/")

    async def event_source():
        # Replay nothing — we start from tip; background will emit from tip+1
        async for event in service.subscribe(session_id, from_sequence=tip):
            event_run_id = str(getattr(event, "run_id", "") or "")
            if not _belongs_to_run(event_run_id):
                continue
            yield _event_to_sse_dict(event)
            # Stop on this run's terminal event (helper runs contain "/").
            typ = str(getattr(event, "type", ""))
            if typ in ("run_finished", "error") and "/" not in event_run_id:
                break
            # Also stop if background done and we've drained events up to its finish
            if background.done() and typ in ("run_finished", "error"):
                # Give a moment for final events to flush
                await asyncio.sleep(0.05)
                # If no new events arrive quickly, the loop will idle; break on background done
                # We don't break immediately to avoid cutting off the receipt event
                # Check if the terminal event was already yielded.
                if typ == "run_finished":
                    break
        # Ensure background is awaited (propagate exception if any, but don't crash stream)
        try:
            result = await background
        except Exception as exc:  # noqa: BLE001 - background errors are already represented in events
            log.debug("Native background task ended with error: %s", exc)
        else:
            if result is not None:
                yield _run_receipt_to_sse_dict(result)

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

    # Peek the run id the resume will continue BEFORE registering anything,
    # then pin it: the resumed work emits under exactly this id, so the
    # cancellation below names the right run. Resumes of one session are
    # serialized — a second racing resume would otherwise continue the same
    # interrupted run id with a second loop on one transcript.
    try:
        run_id = await service.peek_resume_run_id(session_id)
    except KeyError:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found") from None
    cancellation = Cancellation()
    _register_cancel(request, session_id, run_id, cancellation)
    async with _resume_lock(request, session_id):
        try:
            result = await service.resume_run(
                session_id, limits=limits, cancellation=cancellation, run_id=run_id
            )
            try:
                lf = getattr(service.runtime.monitoring, "langfuse_client", None)
                if lf is not None:
                    lf.flush()
            except Exception as exc:  # noqa: BLE001 - flushing is best effort
                log.debug("native Langfuse flush failed: %s", exc)
        finally:
            _unregister_cancel(request, session_id, run_id)

    return JSONResponse(content=RunResponse.from_native(result).model_dump(mode="json"))


@router.post("/{session_id}/cancel", status_code=202)
async def cancel_run(
    session_id: str,
    request: Request,
    service: NativeServiceDep,
    run_id: str | None = None,
):
    """Cancel an in-flight native run.

    ``run_id`` names exactly one run: cancelling it never touches another
    run in the same session, and cleanup removes only that run's registry
    entry. Without ``run_id`` every in-flight run of the session is
    cancelled and the response names each one that was stopped.
    """
    from fastapi import HTTPException

    db = service.runtime.database
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")

    cancels = _cancel_registry(request)
    if run_id is not None:
        cancellation = cancels.get(run_id)
        if cancellation is None:
            # Unknown or already-finished run — idempotent 202 with hint.
            return JSONResponse(
                status_code=202,
                content={"session_id": session_id, "run_id": run_id, "cancelled": False, "reason": "no active run"},
            )
        cancellation.cancel()
        return JSONResponse(
            status_code=202,
            content={"session_id": session_id, "run_id": run_id, "cancelled": True},
        )

    run_ids = sorted(_cancel_index(request).get(session_id, set()))
    cancelled_runs: list[str] = []
    for active_run_id in run_ids:
        cancellation = cancels.get(active_run_id)
        if cancellation is None:
            continue
        cancellation.cancel()
        cancelled_runs.append(active_run_id)
    if not cancelled_runs:
        # No in-flight run to cancel — idempotent 202 with hint
        return JSONResponse(status_code=202, content={"session_id": session_id, "cancelled": False, "reason": "no active run"})
    return JSONResponse(
        status_code=202,
        content={"session_id": session_id, "cancelled": True, "cancelled_runs": cancelled_runs},
    )

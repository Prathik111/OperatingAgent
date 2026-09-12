"""The ``TaskStreamSocket`` — one run's events over SSE and WebSocket.

Both endpoints subscribe to the same ``EventBroker`` topic, so a client can use
whichever transport suits it and see the identical event sequence (including a
replay of whatever the run already emitted before the client connected). Events
are serialized defensively so a stray non-primitive in a payload degrades to a
string instead of tearing down the stream.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse

from ..errors import TaskNotFound, TaskNotInThread
from ..serialization import event_to_dict, event_to_sse

router = APIRouter(tags=["stream"])


async def _task_event_response(task_id: str, request: Request) -> EventSourceResponse:
    service = request.app.state.task_service
    # Validate before EventSourceResponse sends HTTP headers; errors raised from
    # its body iterator cannot be converted into a 404 after response start.
    await service.get_task(task_id)

    async def event_source():
        async for event in service.stream_task(task_id):
            yield event_to_sse(event)

    return EventSourceResponse(event_source(), ping=15)


@router.get("/threads/{thread_id}/tasks/{task_id}/events")
async def stream_thread_task_events(
    thread_id: str,
    task_id: str,
    request: Request,
) -> EventSourceResponse:
    """Stream a task only when it belongs to the requested thread."""
    await request.app.state.task_service.get_task_in_thread(thread_id, task_id)
    return await _task_event_response(task_id, request)


@router.get("/tasks/{task_id}/events", include_in_schema=False)
async def stream_events(task_id: str, request: Request) -> EventSourceResponse:
    """Compatibility stream for older clients."""
    await request.app.state.task_service.get_task(task_id)
    return await _task_event_response(task_id, request)


async def _stream_ws(websocket: WebSocket, task_id: str) -> None:
    """WebSocket stream of one task's run events."""
    service = websocket.app.state.task_service
    await websocket.accept()
    try:
        async for event in service.stream_task(task_id):
            await websocket.send_json(event_to_dict(event))
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/threads/{thread_id}/tasks/{task_id}")
async def stream_thread_ws(websocket: WebSocket, thread_id: str, task_id: str) -> None:
    try:
        await websocket.app.state.task_service.get_task_in_thread(thread_id, task_id)
    except (TaskNotFound, TaskNotInThread):
        await websocket.close(code=4404)
        return
    await _stream_ws(websocket, task_id)


@router.websocket("/ws/tasks/{task_id}")
async def stream_ws(websocket: WebSocket, task_id: str) -> None:
    """Compatibility WebSocket stream for older clients."""
    try:
        await websocket.app.state.task_service.get_task(task_id)
    except TaskNotFound:
        await websocket.close(code=4404)
        return
    await _stream_ws(websocket, task_id)

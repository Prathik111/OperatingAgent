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

from ..errors import TaskNotFound
from ..serialization import event_to_dict, event_to_sse

router = APIRouter(tags=["stream"])


@router.get("/tasks/{task_id}/events")
async def stream_events(
    task_id: str,
    request: Request,
) -> EventSourceResponse:
    """Server-Sent Events stream of a task's run events."""
    service = request.app.state.task_service
    await service.get_task(task_id)  # raises TaskNotFound -> 404 before we stream
    broker = request.app.state.broker

    async def event_source():
        async for event in broker.subscribe(task_id):
            yield event_to_sse(event)

    return EventSourceResponse(event_source(), ping=15)


@router.websocket("/ws/tasks/{task_id}")
async def stream_ws(websocket: WebSocket, task_id: str) -> None:
    """WebSocket stream of the same run events; closes with 4404 if unknown."""
    service = websocket.app.state.task_service
    broker = websocket.app.state.broker

    try:
        await service.get_task(task_id)
    except TaskNotFound:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    try:
        async for event in broker.subscribe(task_id):
            await websocket.send_json(event_to_dict(event))
    except WebSocketDisconnect:
        pass

"""Domain exceptions and their HTTP mapping.

The service and repository layers raise these plain exceptions; a single
exception handler (registered by :func:`register_exception_handlers`) turns any
``ApiError`` into a JSON ``{"detail": ...}`` response with the right status
code. Keeping the mapping in one place means the routers stay free of
``HTTPException`` plumbing and the same errors are reusable from non-HTTP call
sites (tests, a future CLI).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Base class for errors that carry an HTTP status and a client-safe detail."""

    status_code: int = 500
    detail: str = "internal error"

    def __init__(self, detail: str | None = None) -> None:
        if detail is not None:
            self.detail = detail
        super().__init__(self.detail)


class TaskNotFound(ApiError):
    status_code = 404

    def __init__(self, task_id: str) -> None:
        super().__init__(f"task '{task_id}' not found")


class ThreadNotFound(ApiError):
    status_code = 404

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"thread '{thread_id}' not found")


class TaskNotInThread(ApiError):
    status_code = 404

    def __init__(self, task_id: str, thread_id: str) -> None:
        super().__init__(f"task '{task_id}' does not belong to thread '{thread_id}'")


class UnknownTrack(ApiError):
    status_code = 400

    def __init__(self, track: str) -> None:
        super().__init__(f"no orchestrator registered for track '{track}'")


class ApprovalNotFound(ApiError):
    status_code = 404

    def __init__(self, request_id: str) -> None:
        super().__init__(f"approval request '{request_id}' not found")


class ApprovalAlreadyResolved(ApiError):
    status_code = 409

    def __init__(self, request_id: str) -> None:
        super().__init__(f"approval request '{request_id}' is already resolved")


class TaskAlreadyRunning(ApiError):
    status_code = 409

    def __init__(self, task_id: str) -> None:
        super().__init__(f"task '{task_id}' already has a run in progress")


class InvalidWorkspace(ApiError):
    status_code = 422

    def __init__(self, workspace: str) -> None:
        super().__init__(f"workspace '{workspace}' must be an existing directory")


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every ``ApiError`` to a JSON response with its status code."""

    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

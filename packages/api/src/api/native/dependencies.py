"""FastAPI dependencies for the native track.

Thin accessors over app.state.native_* set in the lifespan. Kept separate from
the Task-track dependencies so existing tests can override without pulling the
native runtime.
"""

from __future__ import annotations

from fastapi import Request


def get_native_service(request: Request):
    svc = getattr(request.app.state, "native_service", None)
    if svc is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="native service not initialized")
    return svc


def get_native_runtime(request: Request):
    rt = getattr(request.app.state, "native_runtime", None)
    if rt is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="native runtime not initialized")
    return rt


def get_native_database(request: Request):
    rt = get_native_runtime(request)
    return rt.database


def get_native_settings(request: Request):
    return request.app.state.settings

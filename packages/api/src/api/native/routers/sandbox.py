"""Native sandbox status — a live Docker probe for the settings UI.

Probed fresh on every call (``docker info`` + image inspect), never from a
cache, so starting Docker while the app runs flips this to connected on the
next poll with no restart. New sessions then pick up containers on demand via
``ContainerPool.get``; already-cached session runners are untouched.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_native_runtime

router = APIRouter(prefix="/native", tags=["native-sandbox"])


class SandboxContainerRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)


def _container_rows(pool: Any) -> list[dict[str, str]]:
    try:
        rows = pool.list_containers()
    except (AttributeError, TypeError, ValueError):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


@router.get("/sandbox")
async def sandbox_status(
    runtime: Annotated[Any, Depends(get_native_runtime)],
) -> dict[str, Any]:
    pool = getattr(runtime, "sandbox", None)
    if pool is None:
        return {
            "available": False,
            "image": "",
            "status": "sandbox: off - not configured",
            "reason": "sandbox is not configured",
        }
    try:
        available = bool(await pool.probe())
    except Exception as exc:  # noqa: BLE001 - a probe must never 500 the UI
        available = False
        try:
            pool.reason = str(exc) or f"sandbox probe failed ({type(exc).__name__})"
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        status = str(pool.status_line())
    except (AttributeError, TypeError, ValueError):
        status = "sandbox: on" if available else "sandbox: off"
    return {
        "available": available,
        "image": str(getattr(pool, "image", "") or ""),
        "status": status,
        "reason": str(getattr(pool, "reason", "") or ""),
        "containers": _container_rows(pool),
    }


@router.post("/sandbox/containers")
async def create_sandbox_container(
    body: SandboxContainerRequest,
    runtime: Annotated[Any, Depends(get_native_runtime)],
) -> dict[str, Any]:
    """Prewarm the sandbox for an existing Native session."""
    pool = getattr(runtime, "sandbox", None)
    if pool is None:
        raise HTTPException(status_code=409, detail="sandbox is not configured")
    session = await runtime.database.get_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"native session '{body.session_id}' not found")
    workspace = str(getattr(session, "working_directory", ".") or ".")
    runner = await pool.get(body.session_id, workspace)
    if runner is None:
        raise HTTPException(status_code=503, detail=pool.reason or "sandbox container could not be created")
    row = next(
        (item for item in _container_rows(pool) if item.get("session_id") == body.session_id),
        None,
    )
    return row or {
        "session_id": body.session_id,
        "workspace": workspace,
        "container_id": str(getattr(runner, "container_id", "")),
        "image": str(getattr(runner, "image", "") or ""),
        "status": "running",
    }


@router.delete("/sandbox/containers/{session_id}")
async def delete_sandbox_container(
    session_id: str,
    runtime: Annotated[Any, Depends(get_native_runtime)],
) -> dict[str, Any]:
    """Stop all pool-owned containers for one Native session."""
    pool = getattr(runtime, "sandbox", None)
    if pool is None:
        raise HTTPException(status_code=409, detail="sandbox is not configured")
    count = await pool.destroy_session(session_id)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"no active sandbox for session '{session_id}'")
    return {"session_id": session_id, "deleted": count}

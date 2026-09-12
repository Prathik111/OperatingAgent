"""Native sandbox status — a live Docker probe for the settings UI.

Probed fresh on every call (``docker info`` + image inspect), never from a
cache, so starting Docker while the app runs flips this to connected on the
next poll with no restart. New sessions then pick up containers on demand via
``ContainerPool.get``; already-cached session runners are untouched.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ..dependencies import get_native_runtime

router = APIRouter(prefix="/native", tags=["native-sandbox"])


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
            pool.reason = str(exc) or "sandbox probe failed"
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
    }

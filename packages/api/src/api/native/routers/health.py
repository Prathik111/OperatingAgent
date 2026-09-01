"""Native health — session-native liveness."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..dependencies import get_native_runtime, get_native_settings
from ..schemas import NativeHealthResponse

router = APIRouter(prefix="/native", tags=["native-health"])


@router.get("/health", response_model=NativeHealthResponse)
async def native_health(
    runtime: Annotated[object, Depends(get_native_runtime)],
    settings: Annotated[object, Depends(get_native_settings)],
) -> NativeHealthResponse:
    agents = sorted(getattr(runtime, "agents", {}).keys()) if isinstance(getattr(runtime, "agents", None), dict) else []
    models: list[str] = []
    try:
        reg = getattr(runtime, "models", None)
        if reg is not None and hasattr(reg, "_models"):
            models = sorted(reg._models.keys())  # type: ignore[attr-defined]
    except Exception:
        models = []
    backend = getattr(settings, "repository_backend", "memory") or "memory"
    # If a real DSN is set, report postgres even when Task API is still memory
    if getattr(settings, "database_url", None):
        # report which store the native track is actually using
        from agent_native.database import MemoryDatabase

        db = getattr(runtime, "database", None)
        if not isinstance(db, MemoryDatabase):
            backend = "postgres"
    return NativeHealthResponse(status="ok", database=backend, agents=agents, models=models)

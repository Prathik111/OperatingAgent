"""Liveness/readiness probe."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from ..config import ApiSettings
from ..dependencies import get_settings, get_task_service
from ..schemas import HealthResponse
from ..services.task_service import TaskService

router = APIRouter(tags=["health"])

SettingsDep = Annotated[ApiSettings, Depends(get_settings)]
TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: SettingsDep,
    service: TaskServiceDep,
    request: Request,
) -> HealthResponse:
    degraded = list(getattr(request.app.state, "degraded", None) or [])
    return HealthResponse(
        status="degraded" if degraded else "ok",
        repository=settings.repository_backend,
        tracks=service.available_tracks,
        degraded=degraded,
    )

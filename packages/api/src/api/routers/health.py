"""Liveness/readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import ApiSettings
from ..dependencies import get_settings, get_task_service
from ..schemas import HealthResponse
from ..services.task_service import TaskService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: ApiSettings = Depends(get_settings),
    service: TaskService = Depends(get_task_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        repository=settings.repository_backend,
        tracks=service.available_tracks,
    )

"""Conversation thread listing and task history endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_task_service
from ..schemas import TaskResponse, ThreadResponse
from ..services.task_service import TaskService

router = APIRouter(prefix="/threads", tags=["threads"])

TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
Limit = Annotated[int, Query(ge=1, le=500)]
Offset = Annotated[int, Query(ge=0)]


@router.get("", response_model=list[ThreadResponse])
async def list_threads(
    service: TaskServiceDep,
    limit: Limit = 100,
    offset: Offset = 0,
) -> list[ThreadResponse]:
    """Return conversation threads ordered by most recent activity."""
    records = await service.list_threads(limit=limit, offset=offset)
    return [ThreadResponse.from_record(record) for record in records]


@router.get("/{thread_id}/tasks", response_model=list[TaskResponse])
async def list_thread_tasks(
    thread_id: str,
    service: TaskServiceDep,
    limit: Limit = 100,
    offset: Offset = 0,
) -> list[TaskResponse]:
    """Return one thread's tasks ordered newest-first."""
    tasks = await service.list_thread_tasks(
        thread_id,
        limit=limit,
        offset=offset,
    )
    return [
        TaskResponse.from_task(
            task,
            status=run_status.value if run_status is not None else None,
        )
        for task, run_status in tasks
    ]

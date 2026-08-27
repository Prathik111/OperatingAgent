"""The ``TaskRouter`` — submit a goal and read a task back."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from ..dependencies import get_task_service
from ..schemas import CreateTaskRequest, TaskResponse
from ..services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=TaskResponse)
async def create_task(
    body: CreateTaskRequest,
    service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Accept a goal and start a run in the background (returns 202).

    The run status is reported as ``pending`` here; poll ``GET /tasks/{id}`` or
    stream ``/tasks/{id}/events`` to watch it progress.
    """
    task = await service.create_task(
        goal=body.goal,
        track=body.track,
        thread_id=body.thread_id,
        metadata=body.metadata,
    )
    return TaskResponse.from_task(task, status="pending")


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    task, run_status = await service.get_task(task_id)  # raises TaskNotFound -> 404
    return TaskResponse.from_task(
        task, status=run_status.value if run_status is not None else None
    )

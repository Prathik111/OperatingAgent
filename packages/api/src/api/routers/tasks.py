"""The ``TaskRouter`` — submit a goal and read a task back."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from ..dependencies import get_task_service
from ..schemas import CreateTaskRequest, ResumeTaskRequest, TaskResponse
from ..services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])

TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=TaskResponse)
async def create_task(
    body: CreateTaskRequest,
    service: TaskServiceDep,
) -> TaskResponse:
    """Accept a goal and start a run in the background (returns 202).

    The run status is reported as ``pending`` here; use the returned thread id to
    read the task or list the conversation history.
    """
    legacy_thread_id = body.model_extra.get("thread_id") if body.model_extra else None
    task = await service.create_task(
        goal=body.goal,
        track=body.track,
        metadata=body.metadata,
        workspace=body.workspace,
        thread_id=str(legacy_thread_id) if legacy_thread_id else None,
    )
    return TaskResponse.from_task(task, status="pending")


@router.get("/{task_id}", response_model=TaskResponse, include_in_schema=False)
async def get_legacy_task(task_id: str, service: TaskServiceDep) -> TaskResponse:
    """Compatibility lookup; new clients must use the thread-scoped route."""
    task, run = await service.get_task_details(task_id)
    return TaskResponse.from_task(
        task,
        status=run.status.value if run is not None else None,
        run=run,
    )


@router.post("/{task_id}/resume", response_model=TaskResponse)
async def resume_task(
    task_id: str,
    body: ResumeTaskRequest,
    service: TaskServiceDep,
) -> TaskResponse:
    """Resume the task from its latest persisted LangGraph checkpoint."""
    task = await service.resume_task(
        task_id,
        resume_value=body.resume_value,
        checkpoint_id=body.checkpoint_id,
    )
    return TaskResponse.from_task(task, status="pending")

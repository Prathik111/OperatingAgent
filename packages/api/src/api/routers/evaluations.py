"""Read-only evaluation aggregates for the desktop dashboard."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from common.enums import AgentTrack

from ..dependencies import get_task_service
from ..services.task_service import TaskService

router = APIRouter(prefix="/evaluations", tags=["evaluations"])
TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]


class EvaluationDashboardResponse(BaseModel):
    available: bool
    reason: str | None = None
    suites: int = 0
    cases: int = 0
    runs: int = 0
    results: int = 0
    pass_rate: float | None = None
    average_score: float | None = None
    avg_latency_ms: float | None = None
    total_tokens: int | None = None
    total_cost: float | None = None
    tool_success_rate: float | None = None
    metrics: list[dict]
    run_breakdown: list[dict]
    comparison: list[dict] = Field(default_factory=list)
    executions: list[dict] = Field(default_factory=list)
    sources: dict[str, bool]


class EvaluationCaseRequest(BaseModel):
    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    working_directory: str = "."
    expected_output_contains: str = ""
    metadata: dict = Field(default_factory=dict)


class StartEvaluationRequest(BaseModel):
    name: str = Field(default="desktop-suite", min_length=1)
    version: str = Field(default="1", min_length=1)
    tracks: list[AgentTrack] = Field(default_factory=lambda: [AgentTrack.LANGGRAPH])
    cases: list[EvaluationCaseRequest] = Field(min_length=1)


class StartEvaluationResponse(BaseModel):
    evaluation_run_ids: list[str]
    status: str = "started"


@router.get("/dashboard", response_model=EvaluationDashboardResponse)
async def evaluation_dashboard(service: TaskServiceDep) -> EvaluationDashboardResponse:
    data = await service.evaluation_dashboard()
    return EvaluationDashboardResponse.model_validate(data)


@router.post("/runs", response_model=StartEvaluationResponse, status_code=202)
async def start_evaluation(body: StartEvaluationRequest, service: TaskServiceDep) -> StartEvaluationResponse:
    run_ids = await service.start_evaluation(
        name=body.name,
        version=body.version,
        tracks=[track for track in body.tracks],
        cases=[case.model_dump() for case in body.cases],
    )
    return StartEvaluationResponse(evaluation_run_ids=run_ids)

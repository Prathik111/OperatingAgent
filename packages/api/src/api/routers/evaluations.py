"""Read-only evaluation aggregates for the desktop dashboard."""

from __future__ import annotations

from typing import Annotated

from common.enums import AgentTrack
from common.judging import OutputCheck
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

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
    excluded_results: int = 0
    rate_limited_results: int = 0
    pass_rate: float | None = None
    average_score: float | None = None
    avg_latency_ms: float | None = None
    total_tokens: int | None = None
    total_cost: float | None = None
    tool_success_rate: float | None = None
    judge_average: float | None = None
    judge_judged: int = 0
    judge_errors: int = 0
    metrics: list[dict]
    run_breakdown: list[dict]
    comparison: list[dict] = Field(default_factory=list)
    executions: list[dict] = Field(default_factory=list)
    sources: dict[str, bool]


class OutputCheckRequest(BaseModel):
    """One named deterministic check, validated by the shared judge's rules."""

    name: str = Field(min_length=1)
    kind: str
    terms: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    pattern: str = ""
    languages: list[str] = Field(default_factory=list)
    min_blocks: int = 1

    @model_validator(mode="after")
    def _validated_by_judge(self) -> OutputCheckRequest:
        # Reject malformed checks here so a bad request fails with a 422
        # instead of dying later inside the background evaluation run.
        OutputCheck.from_dict(self.model_dump())
        return self


class EvaluationCaseRequest(BaseModel):
    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    working_directory: str = "."
    expected_output_contains: str = ""
    checks: list[OutputCheckRequest] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class StartEvaluationRequest(BaseModel):
    name: str = Field(default="desktop-suite", min_length=1)
    version: str = Field(default="1", min_length=1)
    tracks: list[AgentTrack] = Field(default_factory=lambda: [AgentTrack.LANGGRAPH])
    cases: list[EvaluationCaseRequest] = Field(min_length=1)
    judge_model: str = Field(
        default="",
        description=(
            "Optional LLM judge: a model name from the runtime registry. One "
            "judge instance is shared by every track; scores are recorded "
            "under judge.* metrics, separate from deterministic correctness."
        ),
    )


class StartEvaluationResponse(BaseModel):
    evaluation_run_ids: list[str]
    status: str = "started"


@router.get("/dashboard", response_model=EvaluationDashboardResponse)
async def evaluation_dashboard(service: TaskServiceDep) -> EvaluationDashboardResponse:
    data = await service.evaluation_dashboard()
    return EvaluationDashboardResponse.model_validate(data)


@router.post("/runs", response_model=StartEvaluationResponse, status_code=202)
async def start_evaluation(body: StartEvaluationRequest, service: TaskServiceDep) -> StartEvaluationResponse:
    try:
        run_ids = await service.start_evaluation(
            name=body.name,
            version=body.version,
            tracks=[track for track in body.tracks],
            cases=[case.model_dump() for case in body.cases],
            judge_model=body.judge_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartEvaluationResponse(evaluation_run_ids=run_ids)

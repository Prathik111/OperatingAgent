from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from api.config import ApiSettings
from api.repository.base import TaskRepository
from api.repository.memory import InMemoryTaskRepository
from api.services.approval_gateway import ApprovalGateway, ApprovalRequest
from api.services.event_broker import EventBroker
from api.services.task_service import TaskService
from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RiskLevel, RunStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord


class MetricsOrchestrator:
    async def run(self, task, on_event=None):
        now = datetime.now(UTC)
        await on_event(
            AgentEvent(
                type="llm_call",
                payload=LLMCallRecord(
                    node_name="responder",
                    provider="test",
                    model="scripted",
                    prompt_tokens=7,
                    completion_tokens=3,
                    cost=0.01,
                    started_at=now,
                    finished_at=now,
                ).to_payload(),
            )
        )
        phase_id = str(uuid4())
        step_id = str(uuid4())
        await on_event(AgentEvent(type="phase_entered", payload={
            "id": phase_id, "sequence": 0, "phase": "investigate",
        }))
        await on_event(AgentEvent(type="plan_created", payload={
            "phase_id": phase_id,
            "revision": 0,
            "summary": "inspect",
            "steps": [{"id": step_id, "step_number": 0, "description": "read"}],
        }))
        await on_event(AgentEvent(type="finding_recorded", payload={
            "phase_id": phase_id, "plan_step_id": step_id,
            "description": "port", "detail": "8080",
        }))
        await on_event(AgentEvent(type="verification_recorded", payload={
            "plan_step_id": step_id, "result": "verified", "deterministic": True,
        }))
        await on_event(AgentEvent(type="trace_ref", payload={
            "provider": "langfuse", "trace_id": "trace-test",
        }))
        await on_event(AgentEvent(type="phase_exited", payload={"phase_id": phase_id}))
        await on_event(
            AgentEvent(
                type="tool_call",
                payload=ToolCallRecord(
                    tool_name="read_file",
                    arguments={"path": "README.md"},
                    success=True,
                    output={"text": "ok"},
                    risk_level="safe",
                    started_at=now,
                    finished_at=now,
                ).to_payload(),
            )
        )
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output="done",
            duration_ms=1,
            llm_calls=1,
            tool_calls=1,
            total_tokens=10,
            cost=0.01,
        )


@pytest.mark.asyncio
async def test_memory_repository_satisfies_expanded_protocol():
    repository = InMemoryTaskRepository()
    assert isinstance(repository, TaskRepository)


@pytest.mark.asyncio
async def test_task_service_normalizes_metric_events_and_keeps_streaming():
    repository = InMemoryTaskRepository()
    broker = EventBroker()
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: MetricsOrchestrator()},
        repository=repository,
        broker=broker,
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.NATIVE),
        background=set(),
    )

    task = await service.create_task("measure this")
    await service.wait_idle()
    events = [event async for event in service.stream_task(task.id)]

    run_id = next(iter(repository._runs))
    assert [event.type for event in events] == [
        "llm_call", "phase_entered", "plan_created", "finding_recorded",
        "verification_recorded", "trace_ref", "phase_exited", "tool_call",
    ]
    assert repository.llm_calls_for(run_id)[0].prompt_tokens == 7
    assert repository.tool_calls_for(run_id)[0].tool_name == "read_file"
    run = repository._runs[run_id]
    assert len(run.phases) == len(run.plans) == len(run.findings) == 1
    assert len(run.verifications) == len(run.trace_refs) == 1


@pytest.mark.asyncio
async def test_approval_gateway_persists_request_and_resolution():
    repository = InMemoryTaskRepository()
    task = AgentTask(
        id=str(uuid4()), goal="approve", thread_id=str(uuid4()), track=AgentTrack.NATIVE
    )
    await repository.save_task(task)
    run_id = await repository.create_run(task.id, ApiSettings().build_agent_config(AgentTrack.NATIVE))
    gateway = ApprovalGateway(repository=repository)
    request = ApprovalRequest(
        id=str(uuid4()), task_id=task.id, tool_name="write_file", arguments={},
        risk_level=RiskLevel.REVIEW, run_id=run_id, plan_step_id=str(uuid4()),
    )
    waiter = asyncio.create_task(gateway.request_approval(request))
    await asyncio.sleep(0)
    await gateway.resolve_approval(request.id, True, "approved in test")
    assert await waiter is True
    assert repository._runs[run_id].approvals[0]["status"] == "approved"

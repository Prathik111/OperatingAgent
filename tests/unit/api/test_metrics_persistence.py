from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
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
async def test_evaluation_normalizes_workspace_and_exposes_execution_chat():
    repository = InMemoryTaskRepository()
    orchestrator = MetricsOrchestrator()
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.NATIVE, sandbox_workspace="."),
        background=set(),
    )

    await service.start_evaluation(
        name="workspace-check",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "case-1", "goal": "say hello", "working_directory": "."}],
    )
    await service.wait_idle()

    dashboard = await service.evaluation_dashboard()
    execution = dashboard["executions"][0]
    assert execution["track"] == "native"
    assert execution["goal"] == "say hello"
    assert execution["workspace"] == str(Path.cwd().resolve())


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


class LangGraphToolEventOrchestrator:
    """Emits tool_finished exactly as the LangGraph executor does."""

    async def run(self, task, on_event=None):
        await on_event(
            AgentEvent(
                type="tool_finished",
                payload={
                    "call_id": "call-abc",
                    "step_id": 3,
                    "tool": "read_file",
                    "success": True,
                    "output": "file contents",
                },
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


class LangGraphPlanOrchestrator:
    """Emits plan_created with int step ids and verification_recorded exactly
    as the LangGraph nodes do (step_id int, success bool, no plan_step_id)."""

    async def run(self, task, on_event=None):
        await on_event(AgentEvent(type="plan_created", payload={
            "summary": "read then answer",
            "reasoning": "one read is enough",
            "phase": "investigate",
            "steps": [
                {"id": 0, "description": "read the file", "tool_name": "filesystem_read_file", "arguments": {"path": "README.md"}},
                {"id": 1, "description": "answer", "tool_name": "filesystem_read_file", "arguments": {"path": "README.md"}},
            ],
        }))
        await on_event(AgentEvent(type="verification_recorded", payload={
            "step_id": 0,
            "description": "read the file",
            "tool_name": "filesystem_read_file",
            "success": True,
            "reason": "output matches the request",
        }))
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output="done",
            duration_ms=1,
            llm_calls=1,
            tool_calls=1,
            total_tokens=10,
            cost=0.01,
        )


class ReplanningOrchestrator:
    """Emits plan_created three times (a replanning graph) with int step ids,
    plus a verification after the final plan — exactly the shape that used to
    violate uq_plans_run_revision (revision always 0) and cascade into FK
    errors on verification_results."""

    async def run(self, task, on_event=None):
        for revision in range(3):
            await on_event(AgentEvent(type="plan_created", payload={
                "summary": f"attempt {revision}",
                "reasoning": "retry",
                "phase": "investigate",
                "steps": [
                    {"id": 0, "description": "read", "tool_name": "filesystem_read_file", "arguments": {"path": "a"}},
                    {"id": 1, "description": "answer", "tool_name": "filesystem_read_file", "arguments": {"path": "b"}},
                ],
            }))
        await on_event(AgentEvent(type="verification_recorded", payload={
            "step_id": 0,
            "success": True,
            "reason": "verified on the final plan",
        }))
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
async def test_replanned_runs_get_monotonic_plan_revisions():
    repository = InMemoryTaskRepository()
    service = TaskService(
        orchestrators={AgentTrack.LANGGRAPH: ReplanningOrchestrator()},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.LANGGRAPH),
        background=set(),
    )

    await service.create_task("replan away")
    await service.wait_idle()

    run_id = next(iter(repository._runs))
    run = repository._runs[run_id]
    assert run.status is RunStatus.COMPLETED
    revisions = [plan["revision"] for plan in run.plans]
    assert revisions == [0, 1, 2]
    # Every replan assigns fresh step uuids; the verification must cite the
    # FINAL plan's step 0, not a stale one from an earlier plan.
    final_step_ids = [step["id"] for step in run.plans[-1]["steps"]]
    assert run.verifications[0]["plan_step_id"] == final_step_ids[0]


@pytest.mark.asyncio
async def test_evaluation_tracks_execute_sequentially_not_concurrently():
    order: list[str] = []

    class SlowNativeOrchestrator:
        async def run(self, task, on_event=None):
            await asyncio.sleep(0.05)
            order.append("native")
            return AgentRunResult(
                status=RunStatus.COMPLETED, output="n", duration_ms=1,
                llm_calls=1, tool_calls=0, total_tokens=1,
            )

    class FastLangGraphOrchestrator:
        async def run(self, task, on_event=None):
            order.append("langgraph")
            return AgentRunResult(
                status=RunStatus.COMPLETED, output="l", duration_ms=1,
                llm_calls=1, tool_calls=0, total_tokens=1,
            )

    repository = InMemoryTaskRepository()
    service = TaskService(
        orchestrators={
            AgentTrack.NATIVE: SlowNativeOrchestrator(),
            AgentTrack.LANGGRAPH: FastLangGraphOrchestrator(),
        },
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.NATIVE, sandbox_workspace="."),
        background=set(),
    )

    await service.start_evaluation(
        name="sequential",
        version="1",
        tracks=[AgentTrack.NATIVE, AgentTrack.LANGGRAPH],
        cases=[{"id": "c1", "goal": "g"}],
    )
    await service.wait_idle()

    # If the tracks ran concurrently, the fast langgraph run would finish
    # while native was still sleeping. Sequential execution means native's
    # whole run lands first.
    assert order == ["native", "langgraph"]
    assert all(
        run["results"][0]["success"]
        for run in repository._evaluations.values()
    )


@pytest.mark.asyncio
async def test_plan_step_ids_normalized_and_verification_resolved():
    # Regression 1: int step ids crashed the Postgres insert
    # ("cannot cast type smallint to uuid"); the service now assigns uuids.
    # Regression 2: verification_recorded with step_id/success crashed with
    # KeyError 'plan_step_id'; it now resolves the step and maps the verdict.
    repository = InMemoryTaskRepository()
    service = TaskService(
        orchestrators={AgentTrack.LANGGRAPH: LangGraphPlanOrchestrator()},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.LANGGRAPH),
        background=set(),
    )

    await service.create_task("plan something")
    await service.wait_idle()

    run_id = next(iter(repository._runs))
    run = repository._runs[run_id]
    assert run.status is RunStatus.COMPLETED
    assert len(run.plans) == 1
    plan = run.plans[0]
    assert plan["steps"][0]["step_number"] == 0
    assert plan["steps"][1]["step_number"] == 1
    step_ids = [step["id"] for step in plan["steps"]]
    assert all(isinstance(step_id, str) and step_id for step_id in step_ids)
    assert len(set(step_ids)) == 2
    assert len(run.verifications) == 1
    verification = run.verifications[0]
    assert verification["plan_step_id"] == step_ids[0]
    assert verification["result"] == "verified"


@pytest.mark.asyncio
async def test_verification_for_unknown_step_is_skipped_not_fatal():
    repository = InMemoryTaskRepository()

    class OrphanVerificationOrchestrator:
        async def run(self, task, on_event=None):
            await on_event(AgentEvent(type="verification_recorded", payload={
                "step_id": 99, "success": False, "reason": "orphan",
            }))
            return AgentRunResult(
                status=RunStatus.COMPLETED, output="done", duration_ms=1,
                llm_calls=0, tool_calls=0, total_tokens=0,
            )

    service = TaskService(
        orchestrators={AgentTrack.LANGGRAPH: OrphanVerificationOrchestrator()},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.LANGGRAPH),
        background=set(),
    )
    await service.create_task("orphan verification")
    await service.wait_idle()
    run_id = next(iter(repository._runs))
    run = repository._runs[run_id]
    assert run.status is RunStatus.COMPLETED
    assert run.verifications == []


@pytest.mark.asyncio
async def test_tool_finished_with_correlation_keys_persists_and_completes():
    # Regression: the LangGraph executor's call_id/step_id keys used to crash
    # ToolCallRecord.from_payload, failing the whole graph run.
    repository = InMemoryTaskRepository()
    service = TaskService(
        orchestrators={AgentTrack.LANGGRAPH: LangGraphToolEventOrchestrator()},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.LANGGRAPH),
        background=set(),
    )

    await service.create_task("read something")
    await service.wait_idle()

    run_id = next(iter(repository._runs))
    calls = repository.tool_calls_for(run_id)
    assert [call.tool_name for call in calls] == ["read_file"]
    assert calls[0].success is True
    run = repository._runs[run_id]
    assert run.status is RunStatus.COMPLETED


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

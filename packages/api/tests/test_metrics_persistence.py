from __future__ import annotations

from datetime import datetime, timezone

import pytest

from common.agent import AgentRunResult
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord
from packages.api.src.api.config import ApiSettings
from packages.api.src.api.repository.base import TaskRepository
from packages.api.src.api.repository.memory import InMemoryTaskRepository
from packages.api.src.api.services.approval_gateway import ApprovalGateway
from packages.api.src.api.services.event_broker import EventBroker
from packages.api.src.api.services.task_service import TaskService


class MetricsOrchestrator:
    async def run(self, task, on_event=None):
        now = datetime.now(timezone.utc)
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
    assert [event.type for event in events] == ["llm_call", "tool_call"]
    assert repository.llm_calls_for(run_id)[0].prompt_tokens == 7
    assert repository.tool_calls_for(run_id)[0].tool_name == "read_file"

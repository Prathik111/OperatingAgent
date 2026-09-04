"""Fixtures for the hermetic API unit suite.

Everything here is in-process: an in-memory repository, a real ``EventBroker``
and ``ApprovalGateway``, and a recording orchestrator that emits the same event
shape the real tracks do. The ASGI ``app``/``client`` fixtures drive the routers
through ``httpx.ASGITransport`` with ``dependency_overrides`` and the runtime
collaborators set directly on ``app.state`` — so the lifespan never runs and the
eagerly-constructed LangGraph/``ModelProvider`` is never built. No test here
touches a network, an LLM, or a database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from api import create_app
from api.config import ApiSettings
from api.dependencies import get_approval_gateway, get_settings, get_task_service
from api.repository.memory import InMemoryTaskRepository
from api.services.approval_gateway import ApprovalGateway
from api.services.event_broker import EventBroker
from api.services.task_service import TaskService
from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent
from httpx import ASGITransport


class RecordingOrchestrator:
    """An ``IAgentOrchestrator`` that records the task it ran and emits a fixed
    ``state`` -> ``finished`` sequence, then returns a configurable result — or
    raises before finishing, to exercise the service's failure path.
    """

    def __init__(
        self,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        output: str = "done",
        raise_exc: BaseException | None = None,
    ) -> None:
        self.status = status
        self.output = output
        self.raise_exc = raise_exc
        self.seen: list[AgentTask] = []

    async def run(self, task: AgentTask, on_event=None) -> AgentRunResult:
        self.seen.append(task)
        await self._emit(
            on_event,
            AgentEvent(type="state", payload={"status": "running", "current_step": 0}),
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        await self._emit(
            on_event,
            AgentEvent(type="finished", payload={"status": self.status.value}),
        )
        return AgentRunResult(
            status=self.status,
            output=self.output,
            duration_ms=0.0,
            llm_calls=0,
            tool_calls=0,
            total_tokens=0,
        )

    @staticmethod
    async def _emit(on_event, event: AgentEvent) -> None:
        if on_event is None:
            return
        outcome = on_event(event)
        if outcome is not None and hasattr(outcome, "__await__"):
            await outcome


@pytest.fixture
def settings() -> ApiSettings:
    # NATIVE default so create_task needs no track argument and stays hermetic.
    return ApiSettings(repository_backend="memory", default_track=AgentTrack.NATIVE)


@pytest.fixture
def repository() -> InMemoryTaskRepository:
    return InMemoryTaskRepository()


@pytest.fixture
def broker() -> EventBroker:
    return EventBroker()


@pytest.fixture
def approvals() -> ApprovalGateway:
    return ApprovalGateway()


@pytest.fixture
def orchestrator() -> RecordingOrchestrator:
    return RecordingOrchestrator()


@pytest.fixture
def make_orchestrator():
    """Factory for a ``RecordingOrchestrator`` with custom status/output/raise."""
    return RecordingOrchestrator


@pytest.fixture
def orchestrators(orchestrator: RecordingOrchestrator) -> dict:
    # Both tracks resolve to the recording stub — no real orchestrator is built.
    return {AgentTrack.NATIVE: orchestrator, AgentTrack.LANGGRAPH: orchestrator}


@pytest.fixture
def background() -> set:
    return set()


@pytest.fixture
def task_service(
    orchestrators, repository, broker, settings, background
) -> TaskService:
    return TaskService(
        orchestrators=orchestrators,
        repository=repository,
        broker=broker,
        settings=settings,
        background=background,
    )


@pytest.fixture
def app(settings, task_service, broker, approvals, repository):
    application = create_app(settings)
    # ASGITransport never runs the lifespan, so publish the collaborators the
    # routers read straight off app.state, and override the Depends accessors.
    application.state.task_service = task_service
    application.state.broker = broker
    application.state.approvals = approvals
    application.state.repository = repository
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_task_service] = lambda: task_service
    application.dependency_overrides[get_approval_gateway] = lambda: approvals
    return application


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as async_client:
        yield async_client

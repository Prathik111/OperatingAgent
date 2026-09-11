"""LangGraph settings bridge: allow-all toggle roundtrip.

The real ``LangGraphAgent`` is never constructed here; a stub with the same
surface (``config`` / ``reconfigure`` / ``set_auto_approve_all``) proves the
router wiring — flag application, ordering against model validation, and the
live GET value.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from api import create_app
from api.config import ApiSettings
from api.dependencies import get_settings, get_task_service
from api.repository.memory import InMemoryTaskRepository
from api.services.event_broker import EventBroker
from api.services.task_service import TaskService
from common.enums import AgentTrack
from httpx import ASGITransport

from tests.support.langgraph import build_agent_config


class _StubLangGraphOrchestrator:
    def __init__(self, config) -> None:
        self.config = config
        self._auto_approve_all = False
        self.reconfigured: list = []

    @property
    def auto_approve_all(self) -> bool:
        return self._auto_approve_all

    def set_auto_approve_all(self, enabled: bool) -> bool:
        self._auto_approve_all = bool(enabled)
        return self._auto_approve_all

    async def reconfigure(self, config) -> None:
        self.config = config
        self.reconfigured.append(config)


@pytest.fixture
async def langgraph_client() -> AsyncIterator[tuple[httpx.AsyncClient, _StubLangGraphOrchestrator]]:
    agent = _StubLangGraphOrchestrator(build_agent_config())
    settings = ApiSettings(repository_backend="memory")
    service = TaskService(
        orchestrators={AgentTrack.LANGGRAPH: agent},
        repository=InMemoryTaskRepository(),
        broker=EventBroker(),
        settings=settings,
        background=set(),
    )
    app = create_app(settings)
    app.state.task_service = service
    app.state.settings = settings
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_task_service] = lambda: service
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client, agent


@pytest.mark.regression
async def test_langgraph_settings_auto_approve_roundtrip(langgraph_client) -> None:
    client, agent = langgraph_client

    initial = await client.get("/settings/langgraph")
    assert initial.status_code == 200
    assert initial.json()["auto_approve_all"] is False

    enabled = await client.patch("/settings/langgraph", json={"auto_approve_all": True})
    assert enabled.status_code == 200
    assert enabled.json()["auto_approve_all"] is True
    assert agent.auto_approve_all is True

    current = await client.get("/settings/langgraph")
    assert current.json()["auto_approve_all"] is True

    disabled = await client.patch("/settings/langgraph", json={"auto_approve_all": False})
    assert disabled.json()["auto_approve_all"] is False
    assert agent.auto_approve_all is False


@pytest.mark.regression
async def test_langgraph_settings_model_failure_leaves_toggle_untouched(langgraph_client) -> None:
    client, agent = langgraph_client

    rejected = await client.patch(
        "/settings/langgraph", json={"provider": "bogus", "auto_approve_all": True}
    )
    assert rejected.status_code == 422
    assert agent.auto_approve_all is False
    assert (await client.get("/settings/langgraph")).json()["auto_approve_all"] is False

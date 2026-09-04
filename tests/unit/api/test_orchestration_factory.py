"""Orchestrator construction and shared-track dispatch."""

from __future__ import annotations

import pytest
from agent_langgraph.orchestrator.langgraph_agent import LangGraphAgent
from agent_native.config import AgentConfig
from agent_native.database import MemoryDatabase
from agent_native.service import AgentRuntime, AgentService
from api.config import ApiSettings
from api.orchestration.factory import UnavailableOrchestrator, build_orchestrators
from api.orchestration.native import NativeAgentOrchestrator
from api.services.approval_gateway import ApprovalGateway
from common.agent import AgentTask
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent

from tests._scripted import ScriptedProvider, scripted_registry, text_event


def _task() -> AgentTask:
    return AgentTask(id="1", goal="hi", thread_id="t", track=AgentTrack.NATIVE)


def test_native_track_is_unavailable_without_native_service():
    orchestrators = build_orchestrators(ApiSettings(repository_backend="memory"))
    assert isinstance(orchestrators[AgentTrack.NATIVE], UnavailableOrchestrator)
    # The langgraph track is always registered (real or degraded), never missing.
    assert AgentTrack.LANGGRAPH in orchestrators


def test_langgraph_degrades_for_unsupported_provider():
    settings = ApiSettings(
        repository_backend="memory", llm_provider="nonexistent-provider"
    )
    orchestrators = build_orchestrators(settings)
    # ModelProvider raises for an unknown provider -> guarded into a degraded stub.
    assert isinstance(orchestrators[AgentTrack.LANGGRAPH], UnavailableOrchestrator)


def test_invalid_agent_configuration_propagates_before_track_degradation():
    settings = ApiSettings(repository_backend="memory", llm_timeout_seconds=0)

    with pytest.raises(ValueError, match="llm.timeout_seconds must be positive"):
        build_orchestrators(settings)


def test_langgraph_receives_api_approval_gateway(monkeypatch):
    import agent_langgraph.orchestrator.langgraph_agent as agent_module

    monkeypatch.setattr(agent_module, "ModelProvider", lambda config: object())
    approvals = ApprovalGateway()
    orchestrators = build_orchestrators(
        ApiSettings(repository_backend="memory"),
        approval_handler=approvals,
    )

    agent = orchestrators[AgentTrack.LANGGRAPH]
    assert isinstance(agent, LangGraphAgent)
    assert agent._approval_handler is approvals


def test_langgraph_receives_stdio_gateway_settings(monkeypatch):
    import agent_langgraph.orchestrator.langgraph_agent as agent_module

    monkeypatch.setattr(agent_module, "ModelProvider", lambda config: object())
    orchestrators = build_orchestrators(
        ApiSettings(
            repository_backend="memory",
            mcp_gateway_command="custom-python",
            mcp_gateway_args=("-m", "custom_gateway"),
        )
    )

    agent = orchestrators[AgentTrack.LANGGRAPH]
    assert isinstance(agent, LangGraphAgent)
    transport = agent._tool_registry._mcp._client.transport
    assert transport.command == "custom-python"
    assert transport.args == ["-m", "custom_gateway"]


async def test_native_track_runs_agent_service_and_reuses_thread_session(monkeypatch):
    async def no_mcp(*args, **kwargs):
        return []

    import api.native.runtime as native_runtime

    monkeypatch.setattr(native_runtime, "attach_mcp_tools", no_mcp)
    provider = ScriptedProvider([text_event("native answer")])
    runtime = AgentRuntime(
        database=MemoryDatabase(),
        model_registry=scripted_registry(provider),
        agents=[AgentConfig(name="build", model="scripted-1")],
    )
    service = AgentService(runtime)
    orchestrator = NativeAgentOrchestrator(service)
    events: list[AgentEvent] = []

    first = await orchestrator.run(_task(), on_event=events.append)
    second = await orchestrator.run(_task(), on_event=events.append)

    assert first.status is RunStatus.COMPLETED
    assert second.status is RunStatus.COMPLETED
    assert first.output == "native answer"
    assert second.output == "native answer"
    session = await runtime.database.get_session("t")
    assert session is not None
    conversation = await runtime.database.load_conversation("t")
    assert [message.role.value for message in conversation.messages].count("user") == 2
    assert any(event.type == "finished" for event in events)
    assert len(provider.requests) == 2


async def test_unavailable_orchestrator_run_fails_cleanly():
    events: list[AgentEvent] = []
    result = await UnavailableOrchestrator("langgraph", "no model").run(
        _task(), on_event=lambda e: events.append(e)
    )
    assert result.status == RunStatus.FAILED
    assert "error" in [e.type for e in events]
    assert result.metadata.get("error")

"""Tests for the remaining ``common`` dataclasses and interfaces.

Covers ``common.tools``, ``common.agent``, ``common.events``,
``common.exceptions`` and the ``common.interfaces`` protocols. These are the
shared vocabulary both agent tracks speak, so the tests focus on defaults,
mutability, the exception hierarchy, and structural-typing conformance.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus
from common.events import (
    AgentEvent,
    AgentFinished,
    PlanningStarted,
    ToolFinished,
    ToolStarted,
)
from common.exceptions import (
    AgentException,
    PlanningException,
    ToolExecutionException,
    VerificationException,
)
from common.interfaces import IAgentOrchestrator, IMCPClient
from common.tools import ToolCallRequest, ToolCallResult, ToolInfo, ToolSchema

# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_tool_call_result_error_defaults_none() -> None:
    result = ToolCallResult(success=True, output="ok")
    assert result.error is None


def test_tool_info_risk_defaults_safe() -> None:
    info = ToolInfo(
        name="read_file",
        description="reads",
        schema=ToolSchema(input_schema={}, output_schema={}),
    )
    assert info.risk_level == "safe"


def test_tool_dataclasses_are_mutable_slotted() -> None:
    """These are plain (non-frozen) slotted dataclasses: assignable, but only
    to declared fields."""
    request = ToolCallRequest(tool_name="t", arguments={})
    request.tool_name = "u"
    assert request.tool_name == "u"
    with pytest.raises(AttributeError):
        request.undeclared = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------


def test_agent_task_defaults() -> None:
    task = AgentTask(id="t1", goal="do", thread_id="th", track=AgentTrack.LANGGRAPH)
    assert task.metadata == {}
    assert isinstance(task.created_at, datetime)


def test_agent_task_metadata_is_per_instance() -> None:
    a = AgentTask(id="a", goal="g", thread_id="t", track=AgentTrack.NATIVE)
    b = AgentTask(id="b", goal="g", thread_id="t", track=AgentTrack.NATIVE)
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_agent_run_result_defaults() -> None:
    result = AgentRunResult(
        status=RunStatus.COMPLETED,
        output="done",
        duration_ms=12.5,
        llm_calls=1,
        tool_calls=2,
        total_tokens=10,
    )
    assert result.cost == 0.0
    assert result.metadata == {}


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_cls", [PlanningStarted, ToolStarted, ToolFinished, AgentFinished]
)
def test_event_subclasses_carry_type_and_payload(event_cls: type[AgentEvent]) -> None:
    event = event_cls(type="x", payload={"k": 1})
    assert isinstance(event, AgentEvent)
    assert event.type == "x"
    assert event.payload == {"k": 1}


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls", [PlanningException, ToolExecutionException, VerificationException]
)
def test_exception_hierarchy(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, AgentException)
    assert issubclass(AgentException, Exception)
    with pytest.raises(AgentException):
        raise exc_cls("boom")


# ---------------------------------------------------------------------------
# interfaces (runtime-checkable structural typing)
# ---------------------------------------------------------------------------


class _Orchestrator:
    async def run(self, task, on_event=None):
        return None


class _MCPClient:
    async def list_tools(self):
        return []

    async def call_tool(self, request):
        return None


def test_protocol_conformance_is_structural() -> None:
    """A duck-typed object satisfies the Protocol without inheriting it."""
    orchestrator: IAgentOrchestrator = _Orchestrator()
    client: IMCPClient = _MCPClient()
    assert hasattr(orchestrator, "run")
    assert hasattr(client, "list_tools") and hasattr(client, "call_tool")

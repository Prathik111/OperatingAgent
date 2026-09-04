"""Tests for the planner node (``agent_langgraph.nodes.planner``)."""

from __future__ import annotations

import pytest
from agent_langgraph.graph.state import AgentPlan
from agent_langgraph.nodes.planner import PlannerNode, planner_function
from common.enums import TaskStatus
from common.exceptions import PlanningException
from langchain_core.messages import SystemMessage

from tests.support.langgraph import (
    StubModel,
    StubToolRegistry,
    build_context,
    build_runtime,
    make_state,
    make_tool_info,
)


async def test_planner_node_returns_plan_and_resets_pointer(agent_config, stub_model) -> None:
    runtime = build_runtime(build_context(agent_config, model=stub_model))
    delta = await PlannerNode(make_state(), runtime)

    assert isinstance(delta["plan"], AgentPlan)
    assert delta["current_step"] == 0
    assert delta["status"] is TaskStatus.PLANNING


async def test_planner_node_raises_without_goal(agent_config, stub_model) -> None:
    runtime = build_runtime(build_context(agent_config, model=stub_model))
    with pytest.raises(PlanningException):
        await PlannerNode(make_state(goal=""), runtime)


async def test_planner_uses_structured_output_with_json_schema(agent_config, stub_model) -> None:
    runtime = build_runtime(build_context(agent_config, model=stub_model))
    await planner_function("goal", [], runtime)

    schema, method = stub_model.structured_calls[0]
    assert schema is AgentPlan
    assert method == "json_schema"


async def test_planner_prompt_includes_available_tools_hint(agent_config) -> None:
    model = StubModel()
    registry = StubToolRegistry(tools=[make_tool_info("echo_tool", "echoes text")])
    runtime = build_runtime(build_context(agent_config, model=model, tool_registry=registry))

    await planner_function("goal", [], runtime)

    system_message = model.structured_handles[0].invocations[0][0]
    assert isinstance(system_message, SystemMessage)
    assert "You are the planner." in system_message.content
    assert "echo_tool: echoes text" in system_message.content


async def test_planner_prompt_has_no_hint_when_no_tools(agent_config) -> None:
    model = StubModel()
    runtime = build_runtime(
        build_context(agent_config, model=model, tool_registry=StubToolRegistry(tools=[]))
    )
    await planner_function("goal", [], runtime)

    system_message = model.structured_handles[0].invocations[0][0]
    assert system_message.content.startswith("You are the planner.")
    assert "Available tools" not in system_message.content


async def test_planner_tolerates_tool_listing_failure(agent_config) -> None:
    """If the registry can't be reached, planning proceeds without a hint."""
    model = StubModel()
    registry = StubToolRegistry(list_error=RuntimeError("gateway down"))
    runtime = build_runtime(build_context(agent_config, model=model, tool_registry=registry))

    plan = await planner_function("goal", [], runtime)
    assert isinstance(plan, AgentPlan)
    system_message = model.structured_handles[0].invocations[0][0]
    assert system_message.content.startswith("You are the planner.")
    assert "Available tools" not in system_message.content


async def test_planner_wraps_model_error_in_planning_exception(agent_config) -> None:
    model = StubModel(structured_error=RuntimeError("provider exploded"))
    runtime = build_runtime(build_context(agent_config, model=model))

    with pytest.raises(PlanningException) as excinfo:
        await planner_function("goal", [], runtime)
    assert "provider exploded" in str(excinfo.value)


async def test_planner_forwards_prior_messages(agent_config) -> None:
    """Prior conversation messages sit between the system prompt and the new
    human 'generate a plan' turn."""
    from langchain_core.messages import HumanMessage

    model = StubModel()
    runtime = build_runtime(build_context(agent_config, model=model))
    prior = [HumanMessage(content="earlier turn")]

    await planner_function("my goal", prior, runtime)
    messages = model.structured_handles[0].invocations[0]
    assert messages[1] is prior[0]
    assert "my goal" in messages[-1].content

"""Planner tests - structured output, validation, retry, persistence."""

from __future__ import annotations

import pytest

from agent_native.llm import LLMResponse, ToolCall, Usage
from agent_native.planner import Planner, PlanningError
from agent_native.types import AgentTask, StepKind
from conftest import FakeLLM, executor_tools  # noqa: F401

TASK = AgentTask(id="task-p", goal="build an index page")


def usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5)


def _plan_call(steps: list[dict]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id="p", name="create_plan", arguments={"steps": steps})],
        usage=usage(),
    )


def _plan_json(steps: list[dict]) -> LLMResponse:
    import json

    return LLMResponse(text=json.dumps({"steps": steps}), usage=usage())


GOOD_STEPS = [
    {"description": "create index.html", "kind": "tool", "tool_name": "write_file",
     "check": "file_exists=index.html"},
    {"description": "summarize result", "kind": "analysis"},
]


@pytest.mark.asyncio
async def test_plan_via_tool_call_persists(memory_repo, executor_tools):
    planner = Planner(llm=FakeLLM([_plan_call(GOOD_STEPS)]), repository=memory_repo)
    plan = await planner.plan(TASK, executor_tools)
    assert len(plan.steps) == 2
    assert plan.steps[0].kind == StepKind.TOOL
    assert plan.steps[0].tool_name == "write_file"
    assert plan.steps[0].check == "file_exists=index.html"
    assert plan.steps[1].kind == StepKind.ANALYSIS
    assert plan.steps[1].tool_name is None
    saved = await memory_repo.get_plan(TASK.id)
    assert saved is not None and len(saved.steps) == 2
    assert await memory_repo.get_task(TASK.id) is not None


@pytest.mark.asyncio
async def test_plan_via_json_text(memory_repo, executor_tools):
    planner = Planner(llm=FakeLLM([_plan_json(GOOD_STEPS)]), repository=memory_repo)
    plan = await planner.plan(TASK, executor_tools)
    assert plan.steps[0].tool_name == "write_file"


@pytest.mark.asyncio
async def test_retries_once_then_fails(memory_repo, executor_tools):
    bad = _plan_call([{"description": "x", "kind": "tool", "tool_name": "no_such_tool"}])
    good = _plan_call(GOOD_STEPS)
    planner = Planner(llm=FakeLLM([bad, good]), repository=memory_repo)
    plan = await planner.plan(TASK, executor_tools)
    assert len(plan.steps) == 2
    llm = planner.llm
    assert "Previous attempt was rejected" in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_invalid_plan_raises_after_retries(memory_repo, executor_tools):
    bad = _plan_call([{"description": "x", "kind": "tool", "tool_name": "no_such_tool"}])
    planner = Planner(llm=FakeLLM([bad, bad]), repository=memory_repo)
    with pytest.raises(PlanningError):
        await planner.plan(TASK, executor_tools)
    assert await memory_repo.get_plan(TASK.id) is None


@pytest.mark.asyncio
async def test_empty_plan_raises(memory_repo, executor_tools):
    planner = Planner(llm=FakeLLM([_plan_call([])]), repository=memory_repo)
    with pytest.raises(PlanningError):
        await planner.plan(TASK, executor_tools)


@pytest.mark.asyncio
async def test_analysis_step_rejects_tool_name(memory_repo, executor_tools):
    bad = _plan_call([{"description": "x", "kind": "analysis", "tool_name": "write_file"}])
    planner = Planner(llm=FakeLLM([bad, bad]), repository=memory_repo)
    with pytest.raises(PlanningError):
        await planner.plan(TASK, executor_tools)
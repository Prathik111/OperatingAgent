"""Reflector tests - revision, persistence, and the max_replans bound (#1)."""

from __future__ import annotations

import pytest

from agent_native.events import REPLANNING
from agent_native.llm import LLMResponse, ToolCall, Usage
from agent_native.reflector import Reflector, ReplanBudgetExhausted
from agent_native.types import AgentTask, Plan, PlanStep, StepKind, StepStatus
from conftest import EventSink, FakeLLM, executor_tools  # noqa: F401

TASK = AgentTask(id="task-r", goal="deploy the site")


def _with_plan(steps: list[PlanStep]) -> Plan:
    return Plan(task_id=TASK.id, steps=steps)


def _replan_call(steps: list[dict]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id="r", name="create_plan", arguments={"steps": steps})],
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def old_step() -> PlanStep:
    return PlanStep(id="a", description="run deploy.sh", kind=StepKind.TOOL,
                    tool_name="run_command", check="exit_code=0", status=StepStatus.FAILED)


def new_steps() -> list[dict]:
    return [{"description": "read config", "kind": "tool", "tool_name": "read_file"}]


@pytest.mark.asyncio
async def test_replan_produces_revised_plan(memory_repo, sink: EventSink, executor_tools):
    reflector = Reflector(llm=FakeLLM([_replan_call(new_steps())]), repository=memory_repo,
                          max_replans=3, on_event=sink)
    plan = await reflector.replan(TASK, _with_plan([old_step()]), "exit code 1", executor_tools)
    assert plan.steps[0].tool_name == "read_file"
    assert plan.task_id == TASK.id
    saved = await memory_repo.get_plan(TASK.id)
    assert saved is not None and saved.steps[0].tool_name == "read_file"
    assert REPLANNING in sink.kinds()
    assert reflector.replan_count(TASK.id) == 1


@pytest.mark.asyncio
async def test_replan_budget_exhausted(memory_repo, executor_tools):
    lli = FakeLLM([_replan_call(new_steps())])
    reflector = Reflector(llm=lli, repository=memory_repo, max_replans=2)
    for _ in range(2):
        plan = await reflector.replan(TASK, _with_plan([old_step()]), "fail", executor_tools)
        assert plan is not None
    with pytest.raises(ReplanBudgetExhausted):
        await reflector.replan(TASK, _with_plan([old_step()]), "fail", executor_tools)
    assert reflector.replan_count(TASK.id) == 2


@pytest.mark.asyncio
async def test_invalid_revision_raises_budget_exhausted(memory_repo, executor_tools):
    bad = _replan_call([{"description": "x", "kind": "tool", "tool_name": "missing_tool"}])
    reflector = Reflector(llm=FakeLLM([bad]), repository=memory_repo, max_replans=2)
    with pytest.raises(ReplanBudgetExhausted):
        await reflector.replan(TASK, _with_plan([old_step()]), "fail", executor_tools)
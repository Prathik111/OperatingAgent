"""Full-run integration tests for NativeAgent (spec's required matrix):
happy path, REVIEW approved, REVIEW denied, BLOCKED, verification failure +
replan, max_replans exhaustion."""

from __future__ import annotations

import pytest

from agent_native.agent import NativeAgent
from agent_native.approval import ApprovalGateway
from agent_native.compactor import ContextCompactor
from agent_native.events import (
    AGENT_FINISHED,
    PLANNING_STARTED,
    PLANNING_SUCCEEDED,
    REPLAN_BUDGET_EXHAUSTED,
    REPLANNING,
    RUN_FAILED,
    STEP_FAILED,
    STEP_SUCCEEDED,
    STEP_UNVERIFIABLE,
    TOOL_FINISHED,
    TOOL_STARTED,
)
from agent_native.llm import LLMResponse, ToolCall, Usage
from agent_native.planner import Planner
from agent_native.reflector import Reflector
from agent_native.risk import RiskClassifier
from agent_native.types import (
    AgentTask,
    ApprovalDecision,
    PlanStep,
    RunStatus,
    StepKind,
    ToolCallResult,
)
from agent_native.verifier import Verifier
from agent_native.executor import ReactExecutor
from conftest import EventSink, FakeLLM, FakeMCP, make_tool  # noqa: F401


def plan_call(steps: list[dict]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id="plan", name="create_plan", arguments={"steps": steps})],
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def tool_call(name: str, arguments: dict, cid: str = "tc") -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=cid, name=name, arguments=arguments)],
        usage=Usage(input_tokens=10, output_tokens=5),
    )


class AutoApprove(ApprovalGateway):
    async def request_approval(self, task_id: str, step) -> ApprovalDecision:
        return ApprovalDecision.APPROVED


class Deny(ApprovalGateway):
    async def request_approval(self, task_id: str, step) -> ApprovalDecision:
        return ApprovalDecision.DENIED


def build_agent(
    script: list[LLMResponse],
    mcp_results: dict[str, ToolCallResult],
    sink: EventSink,
    approval: ApprovalGateway | None = None,
    max_replans: int = 3,
):
    llm = FakeLLM(script)
    mcp = FakeMCP(mcp_results)
    for name in ("run_command", "write_file", "format", "read_file"):
        mcp.add_tool(make_tool(name))
    repo = _Repo()
    risk = RiskClassifier(allowlist_net_hosts=None)
    compactor = ContextCompactor(token_budget=1_000_000)
    verifier = Verifier()
    planner = Planner(llm=llm, repository=repo)
    reflector = Reflector(llm=llm, repository=repo, max_replans=max_replans, on_event=sink)
    executor = ReactExecutor(llm=llm, mcp=mcp, risk=risk, verifier=verifier,
                             compactor=compactor, approval=approval, on_event=sink)
    agent = NativeAgent(planner=planner, executor=executor, reflector=reflector,
                        repository=repo, llm=llm, mcp=mcp, risk=risk,
                        verifier=verifier, compactor=compactor, approval=approval,
                        settings=None)
    return agent, mcp, repo


class _Repo:
    def __init__(self) -> None:
        self.tasks = {}
        self.plans = {}
        self.results = []

    async def save_task(self, task) -> None:
        self.tasks[task.id] = task

    async def get_task(self, task_id):
        return self.tasks.get(task_id)

    async def save_plan(self, plan) -> None:
        self.plans[plan.task_id] = plan

    async def get_plan(self, task_id):
        return self.plans.get(task_id)

    async def save_run_result(self, result) -> None:
        self.results.append(result)

    async def list_run_results(self, task_id):
        return [r for r in self.results if r.metadata.get("task_id") == task_id]

    async def close(self) -> None:
        pass


def run_script(*responses: LLMResponse) -> list[LLMResponse]:
    return list(responses)


PLAN_GOOD = [
    {"description": "run the build", "kind": "tool", "tool_name": "run_command",
     "check": "exit_code=0"},
    {"description": "summarize the outcome", "kind": "analysis"},
]
PLAN_WRITE = [
    {"description": "write the file", "kind": "tool", "tool_name": "write_file",
     "check": "exit_code=0"},
]
PLAN_FORMAT = [
    {"description": "format the drive", "kind": "tool", "tool_name": "format",
     "check": "exit_code=0"},
]
PLAN_ANALYSIS = [{"description": "recover with reasoning", "kind": "analysis"}]


@pytest.mark.asyncio
async def test_happy_path(sink: EventSink):
    agent, mcp, repo = build_agent(
        run_script(
            plan_call(PLAN_GOOD),
            tool_call("run_command", {"command": "npm run build"}),
            FakeLLM.text("build succeeded"),
            FakeLLM.text("Final summary"),
        ),
        {"run_command": ToolCallResult(success=True, output="0")},
        sink,
    )
    task = AgentTask(id="t-happy", goal="build the project")
    result = await agent.run(task, on_event=sink)
    assert result.status == RunStatus.COMPLETED
    assert result.replans == 0
    assert result.tool_calls == 1 and result.llm_calls == 2
    kinds = sink.kinds()
    for expected in (PLANNING_STARTED, PLANNING_SUCCEEDED, TOOL_STARTED, TOOL_FINISHED,
                     STEP_SUCCEEDED, STEP_UNVERIFIABLE, AGENT_FINISHED):
        assert expected in kinds
    assert mcp.calls == [("run_command", {"command": "npm run build"})]
    assert len(repo.results) == 1
    assert repo.results[0].status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_review_gate_approved(sink: EventSink):
    agent, mcp, _ = build_agent(
        run_script(
            plan_call(PLAN_WRITE),
            tool_call("write_file", {"path": "/w/a.txt", "content": "x"}),
            FakeLLM.text("written"),
        ),
        {"write_file": ToolCallResult(success=True, output="0")},
        sink,
        approval=AutoApprove(),
    )
    task = AgentTask(id="t-approve", goal="write a file")
    result = await agent.run(task, on_event=sink)
    assert result.status == RunStatus.COMPLETED
    assert mcp.calls == [("write_file", {"path": "/w/a.txt", "content": "x"})]


@pytest.mark.asyncio
async def test_review_gate_denied_triggers_replan(sink: EventSink):
    agent, mcp, _ = build_agent(
        run_script(
            plan_call(PLAN_WRITE),
            tool_call("write_file", {"path": "/w/a.txt", "content": "x"}),
            plan_call(PLAN_ANALYSIS),
            FakeLLM.text("recovered via reasoning"),
        ),
        {"write_file": ToolCallResult(success=False, output=None, error="not reached")},
        sink,
        approval=Deny(),
    )
    task = AgentTask(id="t-deny", goal="write a file")
    result = await agent.run(task, on_event=sink)
    assert result.status == RunStatus.COMPLETED
    assert result.replans == 1
    assert REPLANNING in sink.kinds()
    assert STEP_FAILED in sink.kinds()
    assert mcp.calls == []  # denied call never executed


@pytest.mark.asyncio
async def test_blocked_tool_triggers_replan(sink: EventSink):
    agent, mcp, _ = build_agent(
        run_script(
            plan_call(PLAN_FORMAT),
            tool_call("format", {"drive": "C"}),
            plan_call(PLAN_ANALYSIS),
            FakeLLM.text("used reasoning instead"),
        ),
        {},
        sink,
    )
    task = AgentTask(id="t-block", goal="format a drive")
    result = await agent.run(task, on_event=sink)
    assert result.status == RunStatus.COMPLETED
    assert result.replans == 1
    assert REPLANNING in sink.kinds()
    assert mcp.calls == []


@pytest.mark.asyncio
async def test_verification_failure_triggers_replan(sink: EventSink):
    agent, mcp, _ = build_agent(
        run_script(
            plan_call(PLAN_GOOD),
            tool_call("run_command", {"command": "make"}),
            FakeLLM.text("command produced nonzero exit"),
            plan_call(PLAN_ANALYSIS),
            FakeLLM.text("replanned to analysis"),
        ),
        {"run_command": ToolCallResult(success=True, output="1")},  # exit 1 != 0
        sink,
    )
    task = AgentTask(id="t-verif", goal="make project")
    result = await agent.run(task, on_event=sink)
    assert result.status == RunStatus.COMPLETED
    assert result.replans == 1
    assert STEP_FAILED in sink.kinds()


@pytest.mark.asyncio
async def test_max_replans_exhaustion_is_terminal_failure(sink: EventSink):
    failing_step = {"description": "run the build", "kind": "tool",
                    "tool_name": "run_command", "check": "exit_code=0"}
    script = run_script(
        plan_call([failing_step]),
        tool_call("run_command", {"command": "make"}, "a"),
        FakeLLM.text("failed 1"),
        plan_call([failing_step]),
        tool_call("run_command", {"command": "make"}, "b"),
        FakeLLM.text("failed 2"),
        plan_call([failing_step]),
        tool_call("run_command", {"command": "make"}, "c"),
        FakeLLM.text("failed 3"),
        plan_call([failing_step]),
        tool_call("run_command", {"command": "make"}, "d"),
        FakeLLM.text("failed 4"),
    )
    agent, _mcp, repo = build_agent(
        script,
        {"run_command": ToolCallResult(success=True, output="3")},
        sink,
        max_replans=3,
    )
    task = AgentTask(id="t-exhaust", goal="build with flaky tooling")
    result = await agent.run(task, on_event=sink)
    assert result.status == RunStatus.FAILED
    assert result.replans == 3
    assert result.tool_calls == 4 and result.llm_calls == 8
    assert result.failure_reason and "replan budget exhausted" in result.failure_reason
    kinds = sink.kinds()
    assert REPLAN_BUDGET_EXHAUSTED in kinds
    assert RUN_FAILED in kinds
    assert AGENT_FINISHED in kinds
    assert len(repo.results) == 1
    assert repo.results[0].status == RunStatus.FAILED


@pytest.mark.asyncio
async def test_unknown_exception_maps_to_failed_run(sink: EventSink):
    agent, _mcp, _repo = build_agent(
        run_script(tool_call("read_file", {"path": "/x"})),
        {},
        sink,
    )
    task = AgentTask(id="t-crash", goal="crash me")
    result = await agent.run(task, on_event=sink)
    assert result.status == RunStatus.FAILED
    assert RUN_FAILED in sink.kinds()
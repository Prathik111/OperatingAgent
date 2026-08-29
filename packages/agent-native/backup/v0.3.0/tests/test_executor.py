"""ReactExecutor tests - ReAct loop, risk gate, approval, verification."""

from __future__ import annotations

import pytest

from agent_native.approval import ApprovalGateway
from agent_native.compactor import ContextCompactor
from agent_native.events import (
    STEP_FAILED,
    STEP_SUCCEEDED,
    STEP_UNVERIFIABLE,
    TOOL_FINISHED,
    TOOL_STARTED,
)
from agent_native.executor import ReactExecutor
from agent_native.risk import RiskClassifier
from agent_native.types import (
    AgentTask,
    ApprovalDecision,
    PlanStep,
    StepKind,
    StepOutcomeStatus,
    ToolCallResult,
)
from agent_native.verifier import Verifier
from conftest import EventSink, FakeLLM, FakeMCP, make_tool  # noqa: F401

TASK = AgentTask(id="task-e", goal="do the thing")


def make_executor(
    llm: FakeLLM,
    mcp: FakeMCP,
    sink: EventSink | None = None,
    approval: ApprovalGateway | None = None,
    max_calls: int = 5,
    workspace=None,
) -> ReactExecutor:
    return ReactExecutor(
        llm=llm,
        mcp=mcp,
        risk=RiskClassifier(allowlist_net_hosts=None),
        verifier=Verifier(workspace=workspace),
        compactor=ContextCompactor(token_budget=20000),
        approval=approval,
        max_calls_per_step=max_calls,
        on_event=sink,
    )


def tool_step(check: str = "exit_code=0", tool: str = "run_command") -> PlanStep:
    return PlanStep(id="s1", description="run a command", kind=StepKind.TOOL,
                    tool_name=tool, check=check)


def analysis_step() -> PlanStep:
    return PlanStep(id="s2", description="summarize", kind=StepKind.ANALYSIS)


class AutoApprove(ApprovalGateway):
    def __init__(self, timeout_s: float = 30) -> None:
        super().__init__(timeout_s=timeout_s)

    async def request_approval(self, task_id: str, step) -> ApprovalDecision:
        return ApprovalDecision.APPROVED


class Deny(ApprovalGateway):
    def __init__(self, timeout_s: float = 30) -> None:
        super().__init__(timeout_s=timeout_s)

    async def request_approval(self, task_id: str, step) -> ApprovalDecision:
        return ApprovalDecision.DENIED


@pytest.mark.asyncio
async def test_text_only_analysis_step_succeeds_as_unverifiable(sink: EventSink):
    mcp = FakeMCP()
    executor = make_executor(FakeLLM([FakeLLM.text("the summary is X")]), mcp, sink)
    outcome = await executor.execute_step(TASK, analysis_step(), mcp.tools)
    assert outcome.status == StepOutcomeStatus.SUCCESS
    assert outcome.output == "the summary is X"
    assert STEP_UNVERIFIABLE in sink.kinds()
    assert mcp.calls == []


@pytest.mark.asyncio
async def test_tool_call_then_pass(sink: EventSink):
    mcp = FakeMCP({"run_command": ToolCallResult(success=True, output="0")})
    mcp.add_tool(make_tool("run_command"))
    executor = make_executor(FakeLLM([FakeLLM.tool("run_command", {"command": "echo hi"}, "tc1"),
                                     FakeLLM.text("done")]), mcp, sink)
    step = tool_step()
    outcome = await executor.execute_step(TASK, step, mcp.tools)
    assert outcome.status == StepOutcomeStatus.SUCCESS
    assert mcp.calls == [("run_command", {"command": "echo hi"})]
    assert TOOL_STARTED in sink.kinds() and TOOL_FINISHED in sink.kinds()
    assert STEP_SUCCEEDED in sink.kinds()
    assert step.status.value == "done"


@pytest.mark.asyncio
async def test_verification_failure(sink: EventSink):
    mcp = FakeMCP({"run_command": ToolCallResult(success=False, output="", error="exit 1")})
    mcp.add_tool(make_tool("run_command"))
    executor = make_executor(FakeLLM([FakeLLM.tool("run_command", {"command": "false"}, "tc1"),
                                     FakeLLM.text("it ran")]), mcp, sink)
    outcome = await executor.execute_step(TASK, tool_step(), mcp.tools)
    assert outcome.status == StepOutcomeStatus.VERIFY_FAIL
    assert STEP_FAILED in sink.kinds()


@pytest.mark.asyncio
async def test_blocked_call_aborts_step(sink: EventSink):
    mcp = FakeMCP({})
    mcp.add_tool(make_tool("format"))
    executor = make_executor(FakeLLM([FakeLLM.tool("format", {"drive": "C"}, "tc1")]), mcp, sink)
    outcome = await executor.execute_step(TASK, tool_step(tool="format"), mcp.tools)
    assert outcome.status == StepOutcomeStatus.BLOCKED
    assert mcp.calls == []  # never executed
    assert STEP_FAILED in sink.kinds()


@pytest.mark.asyncio
async def test_review_approved_executes(sink: EventSink):
    mcp = FakeMCP({"write_file": ToolCallResult(success=True, output="0")})
    mcp.add_tool(make_tool("write_file"))
    executor = make_executor(FakeLLM([FakeLLM.tool("write_file", {"path": "/w/a"}, "tc1"),
                                     FakeLLM.text("ok")]), mcp, sink,
                             approval=AutoApprove())
    step = PlanStep(id="s1", description="write", kind=StepKind.TOOL,
                    tool_name="write_file", check="exit_code=0")
    outcome = await executor.execute_step(TASK, step, mcp.tools)
    assert outcome.status == StepOutcomeStatus.SUCCESS
    assert mcp.calls == [("write_file", {"path": "/w/a"})]


@pytest.mark.asyncio
async def test_review_denied_aborts(sink: EventSink):
    mcp = FakeMCP({})
    mcp.add_tool(make_tool("write_file"))
    executor = make_executor(FakeLLM([FakeLLM.tool("write_file", {"path": "/w/a"}, "tc1")]),
                             mcp, sink, approval=Deny())
    step = PlanStep(id="s1", description="write", kind=StepKind.TOOL,
                    tool_name="write_file", check="exit_code=0")
    outcome = await executor.execute_step(TASK, step, mcp.tools)
    assert outcome.status == StepOutcomeStatus.DENIED
    assert mcp.calls == []


@pytest.mark.asyncio
async def test_no_gateway_treats_review_as_denied(sink: EventSink):
    mcp = FakeMCP({})
    mcp.add_tool(make_tool("write_file"))
    executor = make_executor(FakeLLM([FakeLLM.tool("write_file", {"path": "/w/a"}, "tc1")]),
                             mcp, sink, approval=None)
    step = PlanStep(id="s1", description="write", kind=StepKind.TOOL,
                    tool_name="write_file", check="exit_code=0")
    outcome = await executor.execute_step(TASK, step, mcp.tools)
    assert outcome.status == StepOutcomeStatus.DENIED


@pytest.mark.asyncio
async def test_max_calls_exceeded(sink: EventSink):
    # exit code 1 fails the check -> the loop keeps calling the model until the cap
    mcp = FakeMCP({"run_command": ToolCallResult(success=True, output="1")})
    mcp.add_tool(make_tool("run_command"))
    always_tools = FakeLLM([FakeLLM.tool("run_command", {"command": "x"}, "tc-loop")])
    executor = make_executor(always_tools, mcp, sink, max_calls=2)
    outcome = await executor.execute_step(TASK, tool_step(), mcp.tools)
    assert outcome.status == StepOutcomeStatus.MAX_CALLS_EXCEEDED
    assert len(always_tools.calls) == 2


@pytest.mark.asyncio
async def test_history_messages_are_well_formed_for_api(sink: EventSink):
    """After a tool turn the message list must be a valid completion history:
    every assistant tool_calls message immediately followed by its results."""
    mcp = FakeMCP({"run_command": ToolCallResult(success=True, output="0")})
    mcp.add_tool(make_tool("run_command"))
    executor = make_executor(FakeLLM([FakeLLM.tool("run_command", {"command": "x"}, "tc1"),
                                     FakeLLM.text("stop")]), mcp, sink)
    await executor.execute_step(TASK, tool_step(), mcp.tools)
    llm = executor.llm
    last_messages = llm.calls[-1]["messages"]
    i = 0
    while i < len(last_messages):
        if last_messages[i].get("tool_calls"):
            assert i + 1 < len(last_messages)
            assert last_messages[i + 1].get("role") == "tool"
            ids = {tc["id"] for tc in last_messages[i]["tool_calls"]}
            assert last_messages[i + 1]["tool_call_id"] in ids
        i += 1
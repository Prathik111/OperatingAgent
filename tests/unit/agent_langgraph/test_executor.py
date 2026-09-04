"""Tests for the executor node (``agent_langgraph.nodes.executor``).

The executor runs exactly one step per invocation and returns a *state delta*.
The behaviours pinned here: reasoning-only steps, the deterministic human gate,
retry-vs-give-up on the two failure kinds, output truncation, and the invariant
that the executor never advances ``current_step`` (the verifier owns that).
"""

from __future__ import annotations

from agent_langgraph.nodes import executor as executor_module
from agent_langgraph.nodes.executor import MAX_OUTPUT_CHARS, ExecutorNode
from common.enums import RunStatus, TaskStatus
from common.tools import ToolCallResult

from tests.support.langgraph import (
    StubToolRegistry,
    build_agent_config,
    build_context,
    build_runtime,
    make_plan,
    make_state,
    make_step,
)


def run_executor(config, state, *, tool_registry=None, **ctx_kwargs):
    context = build_context(config, tool_registry=tool_registry, **ctx_kwargs)
    return ExecutorNode(state, build_runtime(context))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


async def test_no_plan_fails(agent_config) -> None:
    delta = await run_executor(agent_config, make_state(plan=None))
    assert delta["status"] is TaskStatus.FAILED
    assert "no plan" in delta["last_error"]


async def test_pointer_past_end_routes_to_responding(agent_config) -> None:
    state = make_state(plan=make_plan(make_step(1)), current_step=1)
    delta = await run_executor(agent_config, state)
    assert delta["status"] is TaskStatus.RESPONDING


async def test_iteration_budget_exhausted_fails(make_config) -> None:
    config = make_config(max_iterations=1)
    # Two steps, pointer at index 1 which is >= max_iterations (1).
    state = make_state(plan=make_plan(make_step(1), make_step(2)), current_step=1)
    delta = await run_executor(config, state)
    assert delta["status"] is TaskStatus.FAILED
    assert "iteration budget" in delta["last_error"]


# ---------------------------------------------------------------------------
# Reasoning-only steps
# ---------------------------------------------------------------------------


async def test_reasoning_step_completes_without_tool(agent_config) -> None:
    step = make_step(1, tool_name=None, description="think about it")
    delta = await run_executor(agent_config, make_state(plan=make_plan(step), current_step=0))

    assert delta["status"] is TaskStatus.VERIFYING
    updated = delta["plan"].steps[0]
    assert updated.status is RunStatus.COMPLETED
    assert updated.output == "think about it"


# ---------------------------------------------------------------------------
# Successful tool execution
# ---------------------------------------------------------------------------


async def test_successful_tool_call_routes_to_verifying(agent_config) -> None:
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="done", error=None))
    delta = await run_executor(agent_config, make_state(), tool_registry=registry)

    assert delta["status"] is TaskStatus.VERIFYING
    assert delta["plan"].steps[0].status is RunStatus.COMPLETED
    assert delta["plan"].steps[0].output == "done"
    assert delta["retry_count"] == 0
    assert delta["last_error"] is None
    assert registry.calls == [("echo_tool", {"text": "hello"})]


async def test_successful_call_appends_ai_message(agent_config) -> None:
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="done", error=None))
    delta = await run_executor(agent_config, make_state(), tool_registry=registry)
    messages = delta["messages"]
    assert len(messages) == 1
    assert "done" in messages[0].content


async def test_executor_does_not_advance_pointer(agent_config) -> None:
    """The verifier owns the advance; a successful execute must not touch it."""
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="x", error=None))
    delta = await run_executor(agent_config, make_state(current_step=0), tool_registry=registry)
    assert "current_step" not in delta


async def test_large_output_is_truncated(agent_config) -> None:
    huge = "A" * (MAX_OUTPUT_CHARS + 500)
    registry = StubToolRegistry(default=ToolCallResult(success=True, output=huge, error=None))
    delta = await run_executor(agent_config, make_state(), tool_registry=registry)
    output = delta["plan"].steps[0].output
    assert len(output) < len(huge)
    assert "chars omitted" in output


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


async def test_business_failure_is_not_retried(make_config) -> None:
    """A tool that ran and reported failure (success=False) is a deterministic
    fault: fail immediately, increment retry_count, stay in EXECUTING."""
    config = make_config(retry_attempts=3)
    registry = StubToolRegistry(
        default=ToolCallResult(success=False, output=None, error="bad input")
    )
    delta = await run_executor(config, make_state(retry_count=0), tool_registry=registry)

    assert delta["status"] is TaskStatus.EXECUTING
    assert delta["plan"].steps[0].status is RunStatus.FAILED
    assert delta["retry_count"] == 1
    assert "bad input" in delta["last_error"]
    assert len(registry.calls) == 1  # not retried


async def test_transient_failure_is_retried_then_fails(make_config, monkeypatch) -> None:
    """A raised exception is transient: retried up to retry_attempts+1 times."""
    monkeypatch.setattr(executor_module.asyncio, "sleep", _no_sleep)
    config = make_config(retry_attempts=2)
    registry = _AlwaysRaises()
    delta = await run_executor(config, make_state(retry_count=0), tool_registry=registry)

    assert delta["status"] is TaskStatus.EXECUTING
    assert delta["plan"].steps[0].status is RunStatus.FAILED
    assert registry.calls == 3  # retry_attempts(2) + 1


async def test_transient_failure_recovers_before_budget(make_config, monkeypatch) -> None:
    monkeypatch.setattr(executor_module.asyncio, "sleep", _no_sleep)
    config = make_config(retry_attempts=3)
    registry = _FailsThenSucceeds(failures=2)
    delta = await run_executor(config, make_state(), tool_registry=registry)

    assert delta["status"] is TaskStatus.VERIFYING
    assert registry.calls == 3  # 2 failures + 1 success


async def test_timeout_is_treated_as_transient(make_config, monkeypatch) -> None:
    monkeypatch.setattr(executor_module.asyncio, "sleep", _no_sleep)
    config = make_config(retry_attempts=0)
    registry = _RaisesTimeout()
    delta = await run_executor(config, make_state(), tool_registry=registry)
    assert delta["status"] is TaskStatus.EXECUTING
    assert delta["plan"].steps[0].status is RunStatus.FAILED


# ---------------------------------------------------------------------------
# Human approval gate
# ---------------------------------------------------------------------------


async def test_risky_tool_gate_approved_proceeds(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"approved": True})
    config = build_agent_config(require_human_approval=True)
    step = make_step(1, tool_name="delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="gone", error=None))

    delta = await run_executor(config, make_state(plan=make_plan(step)), tool_registry=registry)
    assert delta["status"] is TaskStatus.VERIFYING
    assert registry.calls == [("delete_file", {"path": "x"})]


async def test_risky_tool_gate_rejected_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        executor_module, "interrupt",
        lambda payload: {"approved": False, "reason": "denied by human"},
    )
    config = build_agent_config(require_human_approval=True)
    step = make_step(1, tool_name="delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="gone", error=None))

    delta = await run_executor(config, make_state(plan=make_plan(step), retry_count=0),
                               tool_registry=registry)
    assert delta["status"] is TaskStatus.EXECUTING
    assert delta["plan"].steps[0].status is RunStatus.FAILED
    assert delta["retry_count"] == 1
    assert "denied by human" in delta["last_error"]
    assert registry.calls == []  # tool never ran


async def test_risky_tool_uses_configured_approval_handler(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "interrupt", _fail_if_called)
    handler = _RecordingApprovalHandler(approved=True)
    config = build_agent_config(require_human_approval=True)
    step = make_step(1, tool_name="filesystem_delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="gone", error=None))

    delta = await run_executor(
        config,
        make_state(plan=make_plan(step)),
        tool_registry=registry,
        approval_handler=handler,
        task_id="task-42",
    )

    assert delta["status"] is TaskStatus.VERIFYING
    assert handler.requests[0].task_id == "task-42"
    assert handler.requests[0].tool_name == "filesystem_delete_file"


async def test_interrupts_disabled_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "interrupt", _fail_if_called)
    config = build_agent_config(
        require_human_approval=True,
        enable_interrupts=False,
    )
    step = make_step(1, tool_name="delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="gone", error=None))

    delta = await run_executor(
        config,
        make_state(plan=make_plan(step)),
        tool_registry=registry,
    )

    assert delta["plan"].steps[0].status is RunStatus.FAILED
    assert "interrupts are disabled" in delta["last_error"]
    assert registry.calls == []


async def test_no_gate_when_approval_disabled(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "interrupt", _fail_if_called)
    config = build_agent_config(require_human_approval=False)
    step = make_step(1, tool_name="delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="ok", error=None))

    delta = await run_executor(config, make_state(plan=make_plan(step)), tool_registry=registry)
    assert delta["status"] is TaskStatus.VERIFYING


async def test_no_gate_when_risk_below_threshold(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "interrupt", _fail_if_called)
    config = build_agent_config(require_human_approval=True, risk_threshold="review")
    # read_file classifies SAFE, below the REVIEW threshold -> no gate.
    step = make_step(1, tool_name="read_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="text", error=None))

    delta = await run_executor(config, make_state(plan=make_plan(step)), tool_registry=registry)
    assert delta["status"] is TaskStatus.VERIFYING


# ---------------------------------------------------------------------------
# Test doubles for tool timing/failure behaviour
# ---------------------------------------------------------------------------


async def _no_sleep(_seconds: float) -> None:
    return None


def _fail_if_called(_payload: object) -> object:
    raise AssertionError("human gate must not trigger here")


class _AlwaysRaises:
    def __init__(self) -> None:
        self.calls = 0

    async def call_by_name(self, tool_name: str, arguments: dict) -> ToolCallResult:
        self.calls += 1
        raise RuntimeError("transport blew up")


class _RecordingApprovalHandler:
    def __init__(self, *, approved: bool) -> None:
        self.approved = approved
        self.requests = []

    async def request_approval(self, request) -> bool:
        self.requests.append(request)
        return self.approved


class _FailsThenSucceeds:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def call_by_name(self, tool_name: str, arguments: dict) -> ToolCallResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        return ToolCallResult(success=True, output="recovered", error=None)


class _RaisesTimeout:
    def __init__(self) -> None:
        self.calls = 0

    async def call_by_name(self, tool_name: str, arguments: dict) -> ToolCallResult:
        self.calls += 1
        raise TimeoutError

"""Risk classification must fail closed (P0-5).

If the classifier throws, times out, or returns a malformed verdict, the step
must enter the human gate at BLOCKED — never run ungated as SAFE. A broken
classifier waving tools through is the failure mode this guards.
"""

from __future__ import annotations

import dataclasses

import pytest
from agent_langgraph.nodes import executor as executor_module
from agent_langgraph.nodes.executor import ExecutorNode, _needs_approval
from common.enums import RiskLevel, RunStatus
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


class _ExplodingClassifier:
    """Raises whatever it is constructed with, on every call."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc if exc is not None else RuntimeError("classifier blew up")

    def classify(self, call):
        raise self.exc


class _GarbageClassifier:
    """Returns something that is not a RiskLevel at all."""

    def classify(self, call):
        return "banana"


def _run(config, state, **ctx_kwargs):
    context = build_context(config, **ctx_kwargs)
    return ExecutorNode(state, build_runtime(context))


def _deny_all(payload):
    return {"approved": False, "reason": "no"}


async def test_classifier_exception_enters_gate_at_blocked(monkeypatch) -> None:
    """An exploding classifier gates the step at BLOCKED; the tool never runs."""
    seen: dict = {}

    def fake_interrupt(payload):
        seen.update(payload)
        return {"approved": False, "reason": "no"}

    monkeypatch.setattr(executor_module, "interrupt", fake_interrupt)
    config = build_agent_config(require_human_approval=True)
    step = make_step(1, tool_name="delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="gone", error=None))

    delta = await _run(
        config,
        make_state(plan=make_plan(step)),
        tool_registry=registry,
        risk_classifier=_ExplodingClassifier(),
    )

    assert seen["risk"] == RiskLevel.BLOCKED.value
    assert seen["tool"] == "delete_file"
    assert registry.calls == []
    assert delta["plan"].steps[0].status is RunStatus.FAILED


async def test_classifier_timeout_enters_gate_at_blocked(monkeypatch) -> None:
    """Timeouts fail closed exactly like any other classifier error."""

    seen: dict = {}

    def fake_interrupt(payload):
        seen.update(payload)
        return {"approved": False, "reason": "no"}

    monkeypatch.setattr(executor_module, "interrupt", fake_interrupt)
    config = build_agent_config(require_human_approval=True)
    step = make_step(1, tool_name="delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="gone", error=None))

    delta = await _run(
        config,
        make_state(plan=make_plan(step)),
        tool_registry=registry,
        risk_classifier=_ExplodingClassifier(TimeoutError()),
    )

    assert seen["risk"] == RiskLevel.BLOCKED.value
    assert registry.calls == []
    assert delta["plan"].steps[0].status is RunStatus.FAILED


async def test_malformed_verdict_enters_gate_at_blocked(monkeypatch) -> None:
    """A non-RiskLevel return is treated as a classification failure."""
    seen: dict = {}

    def fake_interrupt(payload):
        seen.update(payload)
        return {"approved": False, "reason": "no"}

    monkeypatch.setattr(executor_module, "interrupt", fake_interrupt)
    config = build_agent_config(require_human_approval=True)
    step = make_step(1, tool_name="delete_file", arguments={"path": "x"})
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="gone", error=None))

    delta = await _run(
        config,
        make_state(plan=make_plan(step)),
        tool_registry=registry,
        risk_classifier=_GarbageClassifier(),
    )

    assert seen["risk"] == RiskLevel.BLOCKED.value
    assert registry.calls == []
    assert delta["plan"].steps[0].status is RunStatus.FAILED


async def test_needs_approval_returns_blocked_on_failure(agent_config) -> None:
    """Unit-level pin: failure maps to BLOCKED, never SAFE."""
    context = build_context(agent_config, risk_classifier=_ExplodingClassifier())
    step = make_step(1, tool_name="read_file", arguments={"path": "x"})
    assert _needs_approval(context, step) is RiskLevel.BLOCKED


async def test_event_sink_failure_propagates_and_tool_never_runs(agent_config) -> None:
    """Regression (P0-6): a failing event sink must raise, not warn-and-continue.

    The sink is the authoritative execution history. Swallowing its failure
    would let the node report success with holes in the record.
    """
    def boom(event):
        raise RuntimeError("disk on fire")

    context = build_context(agent_config)
    context = dataclasses.replace(context, event_sink=boom)
    registry = StubToolRegistry(default=ToolCallResult(success=True, output="done", error=None))

    with pytest.raises(RuntimeError, match="disk on fire"):
        await ExecutorNode(make_state(), build_runtime(context))
    assert registry.calls == []

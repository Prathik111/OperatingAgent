"""Unit tests for the error handler (``agent_langgraph.nodes.error_handler``).

The handler prepares recovery context without routing or re-counting retries —
``retry_router`` owns routing and the verifier owns the counter. It is the one
node whose signature takes only ``state`` (no runtime).
"""

from __future__ import annotations

import pytest
from agent_langgraph.nodes.error_handler import ErrorHandlerNode
from common.enums import RunStatus, TaskStatus
from langchain_core.messages import SystemMessage

from tests.support.langgraph import make_plan, make_state, make_step


def run_error_handler(state):
    return ErrorHandlerNode(state)


def test_error_handler_appends_diagnostic_for_execution_failure() -> None:
    step = make_step(1, status=RunStatus.FAILED)
    delta = run_error_handler(
        make_state(plan=make_plan(step), current_step=0, last_error="tool crashed", retry_count=1)
    )
    assert delta["status"] is TaskStatus.EXECUTING
    message = delta["messages"][0]
    assert isinstance(message, SystemMessage)
    assert "execution failure" in message.content
    assert "tool crashed" in message.content


def test_error_handler_labels_verification_rejection() -> None:
    step = make_step(1, status=RunStatus.COMPLETED)  # ran but was rejected
    delta = run_error_handler(
        make_state(plan=make_plan(step), current_step=0, last_error="wrong output")
    )
    assert "verification rejection" in delta["messages"][0].content


@pytest.mark.regression
def test_error_handler_does_not_touch_retry_count() -> None:
    """Counting is owned upstream; the handler must not re-count — double
    counting would burn the retry budget twice as fast."""
    step = make_step(1, status=RunStatus.FAILED)
    delta = run_error_handler(
        make_state(plan=make_plan(step), current_step=0, retry_count=2, last_error="x")
    )
    assert "retry_count" not in delta


def test_error_handler_with_no_recoverable_step_fails() -> None:
    delta = run_error_handler(make_state(plan=None, current_step=0, last_error="gone"))
    assert delta["status"] is TaskStatus.FAILED
    assert delta["last_error"] == "gone"


def test_error_handler_defaults_reason_when_absent() -> None:
    delta = run_error_handler(make_state(plan=None, current_step=0, last_error=None))
    assert delta["last_error"] == "unknown error"

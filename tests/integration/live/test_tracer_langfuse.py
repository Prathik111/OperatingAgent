"""Live integration: real Langfuse behind ``Tracer`` and ``observability.client``.

The unit tests assert the *disabled* contract — no credentials means
``Tracer.enabled`` is False, ``callback_handler()`` is None, and spans are
no-ops. That contract is only half the story; this module covers the other half,
which no hermetic test can reach: with real credentials the client initialises,
a callback handler exists, and spans actually export.

Ingestion is asynchronous, so these tests assert on the *client and span* objects
and on ``flush()`` completing — not on reading the trace back out of the API.
The e2e live test is what proves a full trace lands (via the returned trace id).
"""

from __future__ import annotations

import pytest
from agent_langgraph.tracing.tracer import Tracer
from agent_native.monitoring import Monitoring
from common.config import TracingConfig
from observability.client import flush, get_callback_handler, get_client
from observability.settings import LangfuseSettings

from tests.support.live import LANGFUSE_HOST

# ---------------------------------------------------------------------------
# Settings and client
# ---------------------------------------------------------------------------


def test_settings_from_env_are_enabled_and_point_at_the_right_region() -> None:
    settings = LangfuseSettings.from_env()
    assert settings.enabled is True
    assert settings.host == LANGFUSE_HOST


def test_get_client_initialises_a_real_client() -> None:
    client = get_client()
    assert client is not None
    assert type(client).__name__ == "Langfuse"


def test_get_callback_handler_returns_a_langchain_handler() -> None:
    """This handler is what carries LangChain/LangGraph spans into Langfuse.

    Asserted by interface, not class name: the SDK exports it as
    ``CallbackHandler`` but the class is ``LangchainCallbackHandler``.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    handler = get_callback_handler()
    assert handler is not None
    assert isinstance(handler, BaseCallbackHandler)


# ---------------------------------------------------------------------------
# Tracer facade
# ---------------------------------------------------------------------------


def test_tracer_is_enabled_with_real_credentials() -> None:
    tracer = Tracer(TracingConfig(enabled=True, provider="langfuse"))
    assert tracer.enabled is True
    assert tracer.callback_handler() is not None


def test_tracer_respects_the_config_switch_even_with_credentials() -> None:
    """``enabled=False`` must win over available credentials — the config switch
    is how an operator turns tracing off, so it cannot be overridden by env."""
    tracer = Tracer(TracingConfig(enabled=False, provider="langfuse"))
    assert tracer.enabled is False
    assert tracer.callback_handler() is None


@pytest.mark.regression
def test_manual_span_reaches_the_real_sdk() -> None:
    """Regression: ``start_span``/``end_span`` used to call SDK methods that do
    not exist (``client.start_span``, ``span.end(output=...)``). Both failures
    were swallowed by the tracer's never-raise contract, so manual spans silently
    did nothing in production while the unit tests — whose fake implemented the
    wrong surface — stayed green. Only a real client catches this.
    """
    tracer = Tracer(TracingConfig(enabled=True, provider="langfuse"))

    span = tracer.start_span("pytest-live-span", suite="integration")
    assert type(span).__name__ != "_NoOpSpan", (
        "manual span degraded to a no-op against the real SDK"
    )
    tracer.end_span(span, result="ok")

    tracer.flush()  # completes without raising == delivered to the collector


def test_span_contextmanager_wraps_a_real_span() -> None:
    tracer = Tracer(TracingConfig(enabled=True, provider="langfuse"))

    with tracer.span("pytest-live-ctx", suite="integration") as span:
        assert type(span).__name__ != "_NoOpSpan"

    tracer.flush()


def test_tool_observation_context_manager_works_live() -> None:
    """The executor wraps every MCP call in ``observation(as_type="tool")``."""
    tracer = Tracer(TracingConfig(enabled=True, provider="langfuse"))

    with tracer.observation(
        "pytest-live-tool", as_type="tool", input={"text": "hello"}
    ) as span:
        assert type(span).__name__ != "_NoOpSpan"

    flush()


@pytest.mark.regression
def test_native_monitoring_trace_reaches_the_real_sdk() -> None:
    """Native's manual observations use the same documented Langfuse lifecycle."""
    monitoring = Monitoring()
    with monitoring.run_span("native-live-run") as run:
        with monitoring.turn_span(1) as turn:
            turn.set(model="live-test", input_tokens=1, output_tokens=1, cost=0.0)
        run.set(status="finished", turns=1)

    trace_id = monitoring.langfuse_trace_ids.get("native-live-run")
    assert trace_id
    monitoring.flush()

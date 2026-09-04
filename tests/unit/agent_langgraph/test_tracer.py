"""Tests for the LangGraph-track tracing facade (``agent_langgraph.tracing.tracer``).

Two regimes matter: with tracing **disabled** (the default in tests — no
credentials) every method must be an inert no-op so the graph behaves
identically; with tracing **enabled** the facade must delegate to the Langfuse
client and still never let a tracing error escape into graph execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
from agent_langgraph.tracing import tracer as tracer_module
from agent_langgraph.tracing.tracer import Tracer, _NoOpSpan
from common.config import TracingConfig

# ---------------------------------------------------------------------------
# Disabled regime (no credentials -> get_client() is None)
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_disabled_even_when_config_enabled_without_credentials() -> None:
    """config.enabled is necessary but not sufficient; without a client the
    facade stays off."""
    assert Tracer(TracingConfig(enabled=True)).enabled is False


def test_disabled_when_config_disabled() -> None:
    assert Tracer(TracingConfig(enabled=False)).enabled is False


def test_callback_handler_none_when_disabled() -> None:
    assert Tracer(TracingConfig(enabled=True)).callback_handler() is None


@pytest.mark.regression
def test_start_span_returns_noop_when_disabled() -> None:
    tracer = Tracer(TracingConfig(enabled=True))
    span = tracer.start_span("work", key="value")
    assert isinstance(span, _NoOpSpan)
    # No-op span methods accept anything and do nothing.
    assert span.update(level="ERROR") is None
    assert span.end(output="x") is None


@pytest.mark.regression
def test_span_contextmanager_is_safe_when_disabled() -> None:
    tracer = Tracer(TracingConfig(enabled=True))
    with tracer.span("work") as handle:
        assert isinstance(handle, _NoOpSpan)


@pytest.mark.regression
def test_observation_yields_noop_when_disabled() -> None:
    tracer = Tracer(TracingConfig(enabled=True))
    with tracer.observation("tool", as_type="tool", input={"a": 1}) as handle:
        assert isinstance(handle, _NoOpSpan)
        handle.update(output="ignored")  # must not raise


@pytest.mark.regression
def test_flush_is_noop_when_disabled() -> None:
    Tracer(TracingConfig(enabled=True)).flush()  # must not raise


# ---------------------------------------------------------------------------
# Enabled regime (client patched in)
# ---------------------------------------------------------------------------


class FakeSpan:
    """Mirrors the real ``LangfuseSpan`` surface the tracer is allowed to use.

    ``end()`` deliberately takes **no** ``output`` keyword, because the real
    SDK's does not. An earlier version of this fake accepted ``**kwargs`` on
    ``end()``, which let ``end_span`` call ``span.end(output=...)`` and pass here
    while silently failing against real Langfuse — the tracing equivalent of a
    green test over dead code. Keep this signature strict.
    """

    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, end_time: Any = None) -> None:
        self.ended = True


class FakeClient:
    """Stands in for ``langfuse.Langfuse``, exposing only methods it really has."""

    def __init__(self) -> None:
        self.span = FakeSpan()
        self.observation_handle = object()
        self.started: list[tuple[str, Any]] = []

    def start_observation(self, *, name: str, input: Any = None, **kwargs: Any) -> FakeSpan:
        self.started.append((name, input))
        return self.span

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):
        self.started.append(("obs", kwargs))
        yield self.observation_handle


@pytest.fixture
def enabled_tracer(monkeypatch: pytest.MonkeyPatch):
    """A Tracer whose client and callback handler are patched in."""
    client = FakeClient()
    handler = object()
    monkeypatch.setattr(tracer_module, "get_client", lambda: client)
    monkeypatch.setattr(tracer_module, "get_callback_handler", lambda: handler)
    flushed: list[bool] = []
    monkeypatch.setattr(tracer_module, "_flush", lambda: flushed.append(True))
    tracer = Tracer(TracingConfig(enabled=True))
    return tracer, client, handler, flushed


def test_enabled_when_client_present(enabled_tracer) -> None:
    tracer, *_ = enabled_tracer
    assert tracer.enabled is True


def test_callback_handler_returned_when_enabled(enabled_tracer) -> None:
    tracer, _client, handler, _ = enabled_tracer
    assert tracer.callback_handler() is handler


@pytest.mark.regression
def test_start_span_uses_the_sdks_observation_constructor(enabled_tracer) -> None:
    """Must call ``start_observation`` — the name the SDK actually provides."""
    tracer, client, _handler, _ = enabled_tracer
    span = tracer.start_span("do-work", detail="x")
    assert span is client.span
    assert client.started[0] == ("do-work", {"detail": "x"})


@pytest.mark.regression
def test_end_span_attaches_output_then_ends(enabled_tracer) -> None:
    """Output goes through ``update()``; ``end()`` is called with no payload."""
    tracer, client, *_ = enabled_tracer
    tracer.end_span(client.span, result="done")
    assert client.span.updates == [{"output": {"result": "done"}}]
    assert client.span.ended is True


def test_end_span_without_output_still_ends_the_span(enabled_tracer) -> None:
    tracer, client, *_ = enabled_tracer
    tracer.end_span(client.span)
    assert client.span.updates == []
    assert client.span.ended is True


def test_observation_yields_client_handle(enabled_tracer) -> None:
    tracer, client, _handler, _ = enabled_tracer
    with tracer.observation("echo_tool", as_type="tool", input={"t": 1}) as handle:
        assert handle is client.observation_handle


def test_flush_delegates_when_enabled(enabled_tracer) -> None:
    tracer, _client, _handler, flushed = enabled_tracer
    tracer.flush()
    assert flushed == [True]


@pytest.mark.regression
def test_start_span_swallows_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadClient:
        def start_observation(self, **_: Any):
            raise RuntimeError("span backend down")

    monkeypatch.setattr(tracer_module, "get_client", lambda: BadClient())
    tracer = Tracer(TracingConfig(enabled=True))
    assert isinstance(tracer.start_span("x"), _NoOpSpan)  # degraded, not raised


@pytest.mark.regression
def test_observation_swallows_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadClient:
        def start_as_current_observation(self, **_: Any):
            raise RuntimeError("obs backend down")

    monkeypatch.setattr(tracer_module, "get_client", lambda: BadClient())
    tracer = Tracer(TracingConfig(enabled=True))
    with tracer.observation("tool") as handle:
        assert isinstance(handle, _NoOpSpan)

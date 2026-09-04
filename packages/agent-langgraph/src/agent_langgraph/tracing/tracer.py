"""Tracer — LangGraph-track facade over the shared observability package.

The heavy lifting is done by the Langfuse LangChain integration (a
``CallbackHandler`` passed into the graph invocation), which auto-captures
every node, generation, and tool call. This class exists so nodes and
``AgentContext`` have a stable handle for:

- obtaining the callback handler to attach to a run, and
- the occasional *manual* span around non-LLM work that the callback handler
  wouldn't otherwise see.

It never raises and degrades to no-ops when tracing is disabled, so the graph
behaves identically with or without Langfuse configured.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

from common.config import TracingConfig
from observability.client import flush as _flush
from observability.client import get_callback_handler, get_client

log = logging.getLogger(__name__)

type ObservationType = Literal[
    "generation",
    "embedding",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
]


class _NoOpSpan:
    """Span handle used when tracing is off; every method is inert."""

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass


class Tracer:
    """Tracing facade for the LangGraph track.

    ``config.enabled`` gates tracing at the app level; the observability
    package additionally requires credentials to be present, so this is safe
    even when config says enabled but no keys are set.
    """

    def __init__(self, config: TracingConfig) -> None:
        self._config = config
        self._enabled = bool(config.enabled) and get_client() is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def callback_handler(self) -> Any | None:
        """Return the Langfuse CallbackHandler to attach to a graph run.

        Pass into the invocation as ``config={"callbacks": [handler]}`` when
        not None. This is the primary, framework-native tracing path.
        """
        if not self._enabled:
            return None
        return get_callback_handler()

    def start_span(self, name: str, **attributes: Any) -> Any:
        """Begin a manual span; a no-op handle when disabled.

        Prefer the callback handler for LLM/graph work — use manual spans only
        for non-LLM operations the handler cannot observe.
        """
        if not self._enabled:
            return _NoOpSpan()
        client = get_client()
        if client is None:
            return _NoOpSpan()
        try:
            # `start_observation` is the SDK's span constructor (there is no
            # `client.start_span`). Defaults to as_type="span".
            return client.start_observation(name=name, input=attributes or None)
        except Exception as exc:  # noqa: BLE001 - tracing is best effort
            log.debug("start_span failed, continuing without trace: %s", exc)
            return _NoOpSpan()

    def end_span(self, span: Any, **outputs: Any) -> None:
        """End a previously started span; safe on no-op handles.

        The SDK's ``end()`` takes no ``output`` — payload is attached with
        ``update()`` first, then the span is closed.
        """
        try:
            if outputs:
                span.update(output=outputs)
            span.end()
        except Exception as exc:  # noqa: BLE001 - tracing is best effort
            log.debug("end_span failed: %s", exc)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[Any]:
        """Context manager wrapping start/end, ending the span even on error."""
        handle = self.start_span(name, **attributes)
        try:
            yield handle
        finally:
            self.end_span(handle)

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: ObservationType = "span",
        input: Any = None,
        metadata: Any = None,
    ) -> Generator[Any]:
        """Nested, correctly-typed observation for work the handler can't see.

        The CallbackHandler covers everything LangChain/LangGraph executes, but
        direct calls that bypass LangChain — notably MCP tool invocations —
        would otherwise be invisible. Wrapping them here makes them appear in
        the trace tree with the right observation type (``tool``), which also
        feeds the Agent Graph.

        Nests under whatever observation is currently active, including spans
        created by the CallbackHandler, so the hierarchy stays correct.
        """
        client = get_client() if self._enabled else None
        if client is None:
            yield _NoOpSpan()
            return

        try:
            manager = client.start_as_current_observation(
                name=name, as_type=as_type, input=input, metadata=metadata
            )
        except Exception as exc:  # noqa: BLE001 - tracing is best effort
            log.debug("could not start observation %s: %s", name, exc)
            yield _NoOpSpan()
            return

        with manager as handle:
            yield handle

    def flush(self) -> None:
        """Flush buffered traces (delegates to the shared client)."""
        if self._enabled:
            _flush()

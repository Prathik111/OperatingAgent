"""Langfuse client + LangChain callback handler factory.

This is the single place the app initialises Langfuse (v4). Everything else —
the LangGraph track, evaluation, scripts — goes through here so credentials,
masking, environment, and release are configured once and consistently.

Design notes:
- The framework integration (``langfuse.langchain.CallbackHandler``) is used
  rather than manual spans: it captures every LangGraph node, LLM generation
  (model + token usage), and tool call with the correct observation types,
  which manual instrumentation cannot match.
- When credentials are absent the module degrades to a disabled state and
  ``get_callback_handler()`` returns ``None`` — callers simply pass no
  callbacks, so the graph runs identically without tracing infra.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import TYPE_CHECKING, Any
from urllib.request import Request, urlopen

from .masking import mask, mask_otel_spans
from .settings import LangfuseSettings

if TYPE_CHECKING:  # avoid importing heavy SDK at type-check time
    from langfuse import Langfuse

log = logging.getLogger(__name__)

_client: Langfuse | None = None
_initialised = False
# Guards the one-time init so a concurrent caller can never observe
# ``_initialised is True`` while ``_client`` is still unset (see init_tracing).
_init_lock = threading.Lock()


def init_tracing(settings: LangfuseSettings | None = None) -> Langfuse | None:
    """Initialise (once) and return the Langfuse singleton, or None if disabled.

    Idempotent and thread-safe: safe to call from multiple entry points/threads.
    ``_initialised`` is flipped to True only *after* ``_client`` reaches its
    final value (a client, or None when disabled/failed), so a racing
    ``get_client`` never returns a half-initialised ``None``. Import happens
    after settings resolution so env vars are read before the SDK reads them.
    """
    global _client, _initialised

    if _initialised:
        return _client

    with _init_lock:
        # Re-check under the lock: another thread may have finished init while
        # we were blocked acquiring it.
        if _initialised:
            return _client

        resolved = settings or LangfuseSettings.from_env()

        if not resolved.enabled:
            log.info("Langfuse tracing disabled: credentials not set (LANGFUSE_PUBLIC_KEY/SECRET_KEY).")
            _client = None
            _initialised = True
            return None

        client: Langfuse | None
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=resolved.public_key,
                secret_key=resolved.secret_key,
                host=resolved.host,
                environment=resolved.environment,
                release=resolved.release,
                mask=mask,
                mask_otel_spans=mask_otel_spans,
            )
            log.info("Langfuse tracing enabled (env=%s, host=%s).", resolved.environment, resolved.host)
        except Exception as exc:
            log.warning("Langfuse initialisation failed; continuing without tracing: %s", exc)
            client = None

        # Publish the final client before marking initialisation complete.
        _client = client
        _initialised = True
        return _client


def get_client() -> Langfuse | None:
    """Return the initialised client, initialising from env on first use."""
    if not _initialised:
        return init_tracing()
    return _client


def reload_tracing(settings: LangfuseSettings | None = None) -> Langfuse | None:
    """Reload the singleton after runtime settings change."""
    global _client, _initialised
    with _init_lock:
        old = _client
        _client = None
        _initialised = False
        if old is not None:
            try:
                old.shutdown()
            except Exception as exc:  # pragma: no cover - SDK cleanup
                log.debug("Langfuse shutdown during reload failed: %s", exc)
    return init_tracing(settings)


async def fetch_trace_metrics(trace_id: str | None) -> dict[str, Any]:
    """Best-effort Langfuse public API enrichment for one trace.

    The dashboard remains usable when Langfuse is disabled or unreachable.
    Returned values are normalized to the same units as the local run metrics.
    """
    settings = LangfuseSettings.from_env()
    if not trace_id or not settings.enabled:
        return {}

    def request_json(path: str) -> dict[str, Any]:
        token = base64.b64encode(f"{settings.public_key}:{settings.secret_key}".encode()).decode()
        request = Request(f"{settings.host.rstrip('/')}/api/public/{path}", headers={"Authorization": f"Basic {token}"})
        with urlopen(request, timeout=2.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    try:
        trace = await asyncio.to_thread(request_json, f"traces/{trace_id}")
        observations = await asyncio.to_thread(request_json, f"observations?traceId={trace_id}&limit=100")
    except Exception as exc:
        log.debug("Langfuse trace enrichment unavailable: %s", exc)
        return {}
    total_tokens = 0
    total_cost = 0.0
    generations = 0
    for observation in observations.get("data", []) if isinstance(observations.get("data"), list) else []:
        if not isinstance(observation, dict):
            continue
        usage = observation.get("usage") or {}
        if isinstance(usage, dict):
            reported_total = usage.get("totalTokens") or usage.get("total_tokens")
            if reported_total is not None:
                total_tokens += int(reported_total)
            else:
                total_tokens += int(usage.get("input") or usage.get("promptTokens") or 0)
                total_tokens += int(usage.get("output") or usage.get("completionTokens") or 0)
        total_cost += float(observation.get("calculatedTotalCost") or observation.get("cost") or 0)
        if observation.get("type") == "GENERATION":
            generations += 1
    return {
        "latency_ms": float(trace.get("latency", 0) or 0) * 1000 if trace.get("latency") is not None else None,
        "total_tokens": total_tokens or None,
        "cost": total_cost or None,
        "llm_calls": generations or None,
        "trace_id": trace_id,
    }


def get_trace_url(trace_id: str | None) -> str | None:
    """Return the Cloud/self-hosted URL for a trace, when tracing is enabled."""
    if not trace_id:
        return None
    client = get_client()
    if client is None:
        return None
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception as exc:
        log.debug("Langfuse trace URL lookup failed: %s", exc)
        return None


def get_callback_handler() -> Any | None:
    """Return a LangChain/LangGraph ``CallbackHandler``, or None if disabled.

    Pass the result (when not None) in the graph invocation config:
    ``config={"callbacks": [handler]}``. The handler inherits trace attributes
    set via ``propagate_attributes`` / invocation metadata.
    """
    if get_client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:
        log.warning("Could not create Langfuse CallbackHandler: %s", exc)
        return None


def flush() -> None:
    """Flush buffered traces. Call before process exit in short-lived runs."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            log.debug("Langfuse flush failed: %s", exc)


def shutdown() -> None:
    """Flush and shut down the client cleanly."""
    client = get_client()
    if client is not None:
        try:
            client.shutdown()
        except Exception as exc:
            log.debug("Langfuse shutdown failed: %s", exc)

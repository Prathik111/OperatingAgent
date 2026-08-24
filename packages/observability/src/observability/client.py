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

import logging
import threading
from typing import TYPE_CHECKING, Any

from .masking import mask, mask_otel_spans
from .settings import LangfuseSettings

if TYPE_CHECKING:  # avoid importing heavy SDK at type-check time
    from langfuse import Langfuse

log = logging.getLogger(__name__)

_client: "Langfuse | None" = None
_initialised = False
# Guards the one-time init so a concurrent caller can never observe
# ``_initialised is True`` while ``_client`` is still unset (see init_tracing).
_init_lock = threading.Lock()


def init_tracing(settings: LangfuseSettings | None = None) -> "Langfuse | None":
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

        client: "Langfuse | None"
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
        except Exception as exc:  # never let tracing setup break the app
            log.warning("Langfuse initialisation failed; continuing without tracing: %s", exc)
            client = None

        # Publish the final client before marking initialisation complete.
        _client = client
        _initialised = True
        return _client


def get_client() -> "Langfuse | None":
    """Return the initialised client, initialising from env on first use."""
    if not _initialised:
        return init_tracing()
    return _client


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

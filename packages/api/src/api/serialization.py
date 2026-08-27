"""JSON-safe serialization for events and config snapshots.

Two concerns live here:

* **Event serialization** — turning an ``AgentEvent`` into a shape that can be
  put on the wire (an SSE frame or a WebSocket JSON message) without a stray
  ``datetime``/``Enum``/``Path`` in a payload ever crashing the stream.
* **Config snapshots** — turning an ``AgentConfig`` into the per-section JSON
  the ``config_snapshots`` table stores, and a stable content hash over it.
  Secrets (the LLM api key and the checkpoint DSN) are redacted from *both* the
  stored JSON and the hash, so they can never leak through the system of record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from common.config import AgentConfig
from common.events import AgentEvent

#: What a redacted secret is replaced with in a config snapshot.
REDACTED = "__redacted__"


def to_jsonable(value: Any) -> Any:
    """Recursively coerce ``value`` into something ``json.dumps`` accepts.

    ``Enum`` -> its value, ``Path`` -> str, ``datetime``/``date`` -> ISO 8601.
    Mappings and sequences are walked; everything else is returned unchanged
    (a genuinely unserializable leaf will still raise at ``json.dumps`` time,
    which is the correct, loud failure).
    """
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """A JSON-safe ``{"type", "payload"}`` dict — used for WebSocket frames."""
    return {"type": event.type, "payload": to_jsonable(event.payload)}


def event_to_sse(event: AgentEvent) -> dict[str, str]:
    """An SSE frame for ``EventSourceResponse``: the event name plus JSON data.

    ``data`` is always a string (SSE has no other data type); the payload is
    dumped through :func:`to_jsonable` so a non-primitive value degrades to its
    string form instead of raising mid-stream.
    """
    return {
        "event": event.type,
        "data": json.dumps(to_jsonable(event.payload), separators=(",", ":")),
    }


# ---------------------------------------------------------------------------
# Config snapshots
# ---------------------------------------------------------------------------

#: config_snapshots column  ->  AgentConfig attribute it is built from.
_SECTIONS = (
    ("llm_config", "llm"),
    ("execution_config", "execution"),
    ("sandbox_config", "sandbox"),
    ("permissions_config", "permissions"),
    ("checkpoint_config", "checkpoint"),
    ("tracing_config", "tracing"),
    ("behaviour_config", "behaviour"),
    ("prompts_config", "prompts"),
)


def config_to_snapshot(config: AgentConfig) -> dict[str, dict[str, Any]]:
    """Build the redacted, JSON-safe per-section snapshot of an ``AgentConfig``.

    The two secret-bearing fields — ``llm.api_key`` and
    ``checkpoint.connection_string`` (both ``repr=False`` in ``common.config``)
    — are replaced with :data:`REDACTED` when set. Redacting *before* hashing
    means two configs that differ only by a rotated key resolve to the same
    content-addressed snapshot, which is what we want: the key is not part of
    the durable config identity.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for column, attr in _SECTIONS:
        section = to_jsonable(asdict(getattr(config, attr)))
        snapshot[column] = section

    llm = snapshot["llm_config"]
    if llm.get("api_key"):
        llm["api_key"] = REDACTED
    checkpoint = snapshot["checkpoint_config"]
    if checkpoint.get("connection_string"):
        checkpoint["connection_string"] = REDACTED

    return snapshot


def config_content_hash(snapshot: Mapping[str, Any]) -> str:
    """A stable sha256 over the canonical JSON of a (already redacted) snapshot.

    ``sort_keys`` makes the hash independent of dict ordering, so the same
    logical config always addresses the same ``config_snapshots`` row.
    """
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

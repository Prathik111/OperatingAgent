"""Serialization — event framing and redacted, content-addressed config snapshots."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from api.serialization import (
    REDACTED,
    config_content_hash,
    config_to_snapshot,
    event_to_dict,
    event_to_sse,
    to_jsonable,
)
from common.enums import AgentTrack
from common.events import AgentEvent

from tests.support.langgraph import build_agent_config


def test_event_to_dict_shape():
    event = AgentEvent("state", {"status": "running", "current_step": 0})
    assert event_to_dict(event) == {
        "type": "state",
        "payload": {"status": "running", "current_step": 0},
    }


def test_event_to_sse_coerces_non_primitives():
    event = AgentEvent(
        "finished",
        {"track": AgentTrack.NATIVE, "at": datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)},
    )
    frame = event_to_sse(event)

    assert frame["event"] == "finished"
    # enum -> value, datetime -> ISO 8601, and `data` is always a JSON string.
    assert '"track":"native"' in frame["data"]
    assert '"at":"2024-01-02T03:04:05+00:00"' in frame["data"]


def test_to_jsonable_recurses_into_containers():
    out = to_jsonable(
        {"a": [AgentTrack.LANGGRAPH, {"b": datetime(2024, 1, 1, tzinfo=UTC)}]}
    )
    assert out == {"a": ["langgraph", {"b": "2024-01-01T00:00:00+00:00"}]}


def test_config_snapshot_redacts_secrets_and_stringifies_paths():
    config = build_agent_config(connection_string="postgresql://u:pw@h/db")
    snapshot = config_to_snapshot(config)

    # api_key ("test-key") and the DSN never reach the system of record.
    assert snapshot["llm_config"]["api_key"] == REDACTED
    assert snapshot["checkpoint_config"]["connection_string"] == REDACTED
    # Path fields degrade to plain strings so the snapshot is JSON-safe.
    assert isinstance(snapshot["prompts_config"]["planner_prompt"], str)


def test_absent_secrets_are_left_untouched():
    # No DSN configured -> connection_string stays None, not the redaction token.
    snapshot = config_to_snapshot(build_agent_config())
    assert snapshot["checkpoint_config"]["connection_string"] is None


def test_content_hash_is_stable_and_secret_independent():
    base = build_agent_config(connection_string="postgresql://u:pw@h/db")
    rotated = dataclasses.replace(
        base, llm=dataclasses.replace(base.llm, api_key="rotated-key")
    )
    changed = dataclasses.replace(
        base, llm=dataclasses.replace(base.llm, model="a-different-model")
    )

    digest = config_content_hash(config_to_snapshot(base))
    assert len(digest) == 64  # sha256 hexdigest

    # A rotated key must NOT change the content-addressed identity...
    assert config_content_hash(config_to_snapshot(rotated)) == digest
    # ...but a real, non-secret config change must.
    assert config_content_hash(config_to_snapshot(changed)) != digest

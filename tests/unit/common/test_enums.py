"""Tests for ``common.enums``.

These string enums cross package and (per the DB design) persistence
boundaries, so their *values* are part of the contract — a rename would break
stored rows and cross-service messages. The tests pin those values.
"""

from __future__ import annotations

import json

import pytest
from common.enums import (
    AgentTrack,
    RiskLevel,
    RunStatus,
    TaskStatus,
    VerificationResult,
    WorkflowPhase,
)


@pytest.mark.parametrize(
    "enum_cls, expected",
    [
        (AgentTrack, {"native", "langgraph"}),
        (RiskLevel, {"safe", "review", "blocked"}),
        (VerificationResult, {"verified", "not_verified", "skipped"}),
        (
            RunStatus,
            {"created", "pending", "running", "completed", "failed", "interrupted"},
        ),
        (
            TaskStatus,
            {
                "planning", "executing", "verifying", "responding",
                "completed", "failed", "skipped", "interrupted",
            },
        ),
        (WorkflowPhase, {"investigate", "remediate", "complete"}),
    ],
)
def test_enum_values(enum_cls: type, expected: set[str]) -> None:
    assert {member.value for member in enum_cls} == expected


@pytest.mark.parametrize("enum_cls", [AgentTrack, RiskLevel, RunStatus, TaskStatus, VerificationResult, WorkflowPhase])
def test_enums_are_str_subclasses(enum_cls: type) -> None:
    """``str, Enum`` means ``member == "value"`` and JSON serialisation works —
    both relied on across the codebase (e.g. building Langfuse tags)."""
    for member in enum_cls:
        assert isinstance(member, str)
        assert member == member.value
        assert json.dumps(member) == json.dumps(member.value)


def test_lookup_by_value_roundtrips() -> None:
    assert RunStatus("completed") is RunStatus.COMPLETED
    assert TaskStatus("executing") is TaskStatus.EXECUTING
    assert RiskLevel("blocked") is RiskLevel.BLOCKED
    assert WorkflowPhase("remediate") is WorkflowPhase.REMEDIATE


def test_unknown_value_raises() -> None:
    with pytest.raises(ValueError):
        RunStatus("nonexistent")

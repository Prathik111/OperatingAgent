"""Tests for ``common.config`` dataclasses.

These are frozen, slotted config objects. The tests verify the defaults the
rest of the system relies on (e.g. an in-memory-friendly checkpoint backend
string, human approval on by default) and that immutability actually
holds — a mutated config shared across graph invocations would be a subtle bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from common.config import (
    AgentConfig,
    BehaviourConfig,
    CheckpointConfig,
    ExecutionConfig,
    LLMConfig,
    MetadataConfig,
    PromptConfig,
    SandboxConfig,
    ToolPermissionConfig,
    TracingConfig,
)


def test_llm_config_defaults() -> None:
    llm = LLMConfig(provider="openai", model="gpt-x", api_key="k")
    assert (llm.timeout_seconds, llm.temperature, llm.top_p) == (60, 0.0, 1.0)
    assert llm.max_tokens is None
    assert llm.base_url is None


def test_execution_config_defaults() -> None:
    execution = ExecutionConfig()
    assert execution.max_iterations == 20
    assert execution.timeout_seconds == 300
    assert execution.retry_attempts == 2
    assert execution.stream is True
    assert execution.enable_checkpoints is True
    assert execution.enable_interrupts is True


def test_behaviour_config_defaults_are_conservative() -> None:
    """Human approval defaults on — the load-bearing safety gate. The LLM
    semantic verifier defaults off, so the verifier only checks the tool result
    for an error and a tool error routes to the planner. The risk threshold is
    ``review`` — the executor compares risk against exactly this string."""
    behaviour = BehaviourConfig()
    assert behaviour.require_verification is False
    assert behaviour.require_human_approval is True
    assert behaviour.risk_threshold == "review"


def test_checkpoint_and_tracing_defaults() -> None:
    checkpoint = CheckpointConfig()
    assert checkpoint.backend == "postgres"
    assert checkpoint.connection_string is None
    assert checkpoint.namespace == "default"

    tracing = TracingConfig()
    assert tracing.enabled is True


def test_sandbox_defaults() -> None:
    sandbox = SandboxConfig()
    assert sandbox.enabled is True
    assert sandbox.workspace == Path("./workspace")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ExecutionConfig(max_iterations=0), "max_iterations"),
        (lambda: ExecutionConfig(retry_attempts=-1), "retry_attempts"),
        (
            lambda: LLMConfig(provider="openai", model="m", api_key="k", top_p=0),
            "top_p",
        ),
        (lambda: BehaviourConfig(risk_threshold="unknown"), "risk_threshold"),
        (lambda: CheckpointConfig(namespace=""), "namespace"),
    ],
)
def test_invalid_config_values_fail_early(factory, message) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_tool_permissions_default_all_enabled() -> None:
    permissions = ToolPermissionConfig()
    assert all(
        (permissions.file_system, permissions.terminal, permissions.git,
         permissions.search, permissions.knowledge, permissions.memory)
    )


def test_metadata_defaults_are_independent_instances() -> None:
    """``field(default_factory=dict)`` must give each instance its own dicts —
    a shared mutable default would leak tags between configs."""
    first = MetadataConfig()
    second = MetadataConfig()
    first.tags["a"] = "1"
    assert second.tags == {}
    assert first.custom is not second.custom


def test_configs_are_frozen() -> None:
    llm = LLMConfig(provider="openai", model="m", api_key="k")
    with pytest.raises(AttributeError):
        llm.provider = "anthropic"  # type: ignore[misc]


def build_full_config() -> AgentConfig:
    return AgentConfig(
        llm=LLMConfig(provider="stub", model="m", api_key="k"),
        execution=ExecutionConfig(),
        sandbox=SandboxConfig(),
        permissions=ToolPermissionConfig(),
        checkpoint=CheckpointConfig(),
        tracing=TracingConfig(),
        behaviour=BehaviourConfig(),
        prompts=PromptConfig(
            planner_prompt=Path("prompts/planner.txt"),
            verifier_prompt=Path("prompts/verifier.txt"),
            responder_prompt=Path("prompts/responder.txt"),
        ),
    )


def test_agent_config_composes_and_defaults_metadata() -> None:
    config = build_full_config()
    assert isinstance(config.metadata, MetadataConfig)
    assert config.metadata.tags == {}
    # Each complete prompt path is retained for PromptManager.
    assert config.prompts.planner_prompt.parent == Path("prompts")


def test_agent_config_is_frozen() -> None:
    config = build_full_config()
    with pytest.raises(AttributeError):
        config.llm = LLMConfig(provider="x", model="y", api_key="z")  # type: ignore[misc]


@pytest.mark.regression
def test_credentials_are_excluded_from_repr() -> None:
    """``api_key`` and ``connection_string`` carry secrets, so ``repr`` — which
    ends up in logs and tracebacks — must not leak them, even when the config is
    nested inside ``AgentConfig``. Field access must still work normally."""
    api_key = "sk-super-secret-key-value"
    dsn = "postgresql://user:hunter2@db:5432/prod"

    config = AgentConfig(
        llm=LLMConfig(provider="openai", model="gpt-x", api_key=api_key),
        execution=ExecutionConfig(),
        sandbox=SandboxConfig(),
        permissions=ToolPermissionConfig(),
        checkpoint=CheckpointConfig(connection_string=dsn),
        tracing=TracingConfig(),
        behaviour=BehaviourConfig(),
        prompts=PromptConfig(
            planner_prompt=Path("prompts/planner.txt"),
            verifier_prompt=Path("prompts/verifier.txt"),
            responder_prompt=Path("prompts/responder.txt"),
        ),
    )

    # Values remain accessible for the code that actually needs them.
    assert config.llm.api_key == api_key
    assert config.checkpoint.connection_string == dsn

    # ...but the nested repr (and each leaf repr) hides them.
    for text in (repr(config), repr(config.llm), repr(config.checkpoint)):
        assert api_key not in text
        assert "hunter2" not in text
        assert dsn not in text

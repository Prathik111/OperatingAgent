"""Fixtures for the agent-langgraph track.

The reusable stubs and builders live in ``tests.support.langgraph`` so tests
can import them directly; this module only wraps the common ones as fixtures.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_langgraph.runtime.context import AgentContext
from common.config import AgentConfig
from langgraph.runtime import Runtime

from tests.support.langgraph import (
    StubModel,
    StubPromptManager,
    StubToolRegistry,
    build_agent_config,
    build_context,
    build_runtime,
    make_state,
)


@pytest.fixture
def make_config() -> Any:
    """The ``build_agent_config`` factory, as a fixture."""
    return build_agent_config


@pytest.fixture
def agent_config() -> AgentConfig:
    """A hermetic default config: memory checkpointer, tracing off, no gate."""
    return build_agent_config()


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel()


@pytest.fixture
def stub_prompt_manager() -> StubPromptManager:
    return StubPromptManager()


@pytest.fixture
def stub_tool_registry() -> StubToolRegistry:
    return StubToolRegistry()


@pytest.fixture
def make_context() -> Any:
    """The ``build_context`` factory, as a fixture."""
    return build_context


@pytest.fixture
def make_runtime() -> Any:
    """The ``build_runtime`` factory, as a fixture."""
    return build_runtime


@pytest.fixture
def agent_context(
    agent_config: AgentConfig,
    stub_model: StubModel,
    stub_tool_registry: StubToolRegistry,
    stub_prompt_manager: StubPromptManager,
) -> AgentContext:
    """A default context wired to the default stubs."""
    return build_context(
        agent_config,
        model=stub_model,
        tool_registry=stub_tool_registry,
        prompt_manager=stub_prompt_manager,
    )


@pytest.fixture
def runtime(agent_context: AgentContext) -> Runtime[AgentContext]:
    """A ``Runtime`` carrying the default context."""
    return build_runtime(agent_context)


@pytest.fixture
def state_factory() -> Any:
    """The ``make_state`` factory, as a fixture."""
    return make_state

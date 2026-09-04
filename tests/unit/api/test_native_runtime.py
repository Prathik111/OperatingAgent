"""Hermetic tests for native runtime startup wiring."""

from __future__ import annotations

import sys
import types

import pytest
from agent_native.config import AgentConfig
from agent_native.database import MemoryDatabase
from agent_native.service import AgentRuntime
from api.config import ApiSettings
from api.native.runtime import (
    attach_mcp_tools,
    build_native_database,
    wire_native_models,
)


def test_build_native_database_defaults_to_memory() -> None:
    database, pool = build_native_database(ApiSettings(repository_backend="memory"))
    assert isinstance(database, MemoryDatabase)
    assert pool is None


def test_build_native_database_requires_dsn_for_explicit_postgres() -> None:
    with pytest.raises(ValueError, match="DATABASE_URL is not set"):
        build_native_database(ApiSettings(repository_backend="postgres"))


def test_wire_native_models_registers_ollama_defaults_without_groq_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test:11434")
    runtime = AgentRuntime(database=MemoryDatabase())

    registered = wire_native_models(runtime)

    assert "qwen3.5:4b-q4_K_M" in registered
    assert "llama3.1" in registered
    assert runtime.models.get("llama3.1").provider == "ollama"


def test_wire_native_models_registers_configured_ollama_model(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = ApiSettings(llm_provider="ollama", llm_model="my-local-model")
    runtime = AgentRuntime(
        database=MemoryDatabase(),
        agents=[AgentConfig(name="build", model="my-local-model")],
    )

    registered = wire_native_models(runtime, settings=settings)

    assert "my-local-model" in registered
    assert runtime.models.get("my-local-model").provider == "ollama"


async def test_attach_mcp_tools_skips_when_tools_are_already_attached() -> None:
    runtime = AgentRuntime(database=MemoryDatabase())

    class Definition:
        full_name = "filesystem.read_file"

    runtime.tools.all = lambda: [types.SimpleNamespace(definition=Definition())]

    assert await attach_mcp_tools(runtime) == []


async def test_attach_mcp_tools_registers_and_keeps_provider(monkeypatch, tmp_path) -> None:
    runtime = AgentRuntime(database=MemoryDatabase())

    class FakeTool:
        definition = types.SimpleNamespace(full_name="filesystem.read_file")

    class FakeProvider:
        def __init__(self):
            self.root = None

        async def connect(self, *, root):
            self.root = root
            return [FakeTool()]

    fake_module = types.SimpleNamespace(MCPToolProvider=FakeProvider)
    monkeypatch.setitem(sys.modules, "agent_native.tools.mcp_bridge", fake_module)

    tools = await attach_mcp_tools(runtime, working_directory=str(tmp_path))

    assert len(tools) == 1
    assert runtime._mcp_providers[0].root == str(tmp_path.resolve())


async def test_attach_mcp_tools_degrades_when_provider_fails(monkeypatch) -> None:
    runtime = AgentRuntime(database=MemoryDatabase())

    class FailingProvider:
        async def connect(self, *, root):
            raise RuntimeError("gateway unavailable")

    fake_module = types.SimpleNamespace(MCPToolProvider=FailingProvider)
    monkeypatch.setitem(sys.modules, "agent_native.tools.mcp_bridge", fake_module)

    assert await attach_mcp_tools(runtime) == []
    assert not hasattr(runtime, "_mcp_providers")

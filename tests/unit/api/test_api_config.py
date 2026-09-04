from __future__ import annotations

from pathlib import Path

import pytest
from agent_langgraph.runtime.prompt_manager import DEFAULT_PROMPT_DIR, PromptManager
from api.config import ApiSettings
from common.enums import AgentTrack


def test_direct_defaults_match_environment_defaults(monkeypatch) -> None:
    for name in (
        "DATABASE_URL",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "API_CORS_ORIGINS",
        "API_ALLOWED_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)

    direct = ApiSettings()
    sourced = ApiSettings.from_env()

    assert (direct.llm_provider, direct.llm_model, direct.llm_base_url) == (
        sourced.llm_provider,
        sourced.llm_model,
        sourced.llm_base_url,
    )


def test_from_env_populates_runtime_agent_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("LLM_MAX_TOKENS", "900")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "7")
    monkeypatch.setenv("AGENT_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("AGENT_STREAM", "false")
    monkeypatch.setenv("AGENT_ENABLE_INTERRUPTS", "false")
    monkeypatch.setenv("AGENT_PERMISSION_TERMINAL", "false")
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AGENT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("AGENT_CHECKPOINT_NAMESPACE", "tenant-a")
    monkeypatch.setenv("AGENT_REQUIRE_VERIFICATION", "true")
    monkeypatch.setenv("AGENT_REQUIRE_HUMAN_APPROVAL", "false")
    monkeypatch.setenv("MCP_GATEWAY_COMMAND", "custom-python")
    monkeypatch.setenv("MCP_GATEWAY_ARGS", "-m gateway_server --profile local")

    settings = ApiSettings.from_env()
    config = settings.build_agent_config(AgentTrack.LANGGRAPH)

    assert config.llm.model == "qwen3"
    assert config.llm.base_url == "http://ollama.internal:11434"
    assert config.llm.timeout_seconds == 17
    assert config.llm.temperature == 0.4
    assert config.llm.max_tokens == 900
    assert config.execution.max_iterations == 7
    assert config.execution.retry_attempts == 4
    assert config.execution.stream is False
    assert config.execution.enable_interrupts is False
    assert config.permissions.terminal is False
    assert config.sandbox.workspace == tmp_path
    assert config.checkpoint.backend == "memory"
    assert config.checkpoint.namespace == "tenant-a"
    assert config.behaviour.require_verification is True
    assert config.behaviour.require_human_approval is False
    assert settings.mcp_gateway_command == "custom-python"
    assert settings.mcp_gateway_args == (
        "-m",
        "gateway_server",
        "--profile",
        "local",
    )


def test_build_agent_config_uses_exact_prompt_paths_and_track() -> None:
    settings = ApiSettings(
        planner_prompt="one/planner.md",
        verifier_prompt="two/verifier.md",
        responder_prompt="three/responder.md",
    )

    config = settings.build_agent_config(AgentTrack.NATIVE)

    assert config.prompts.planner_prompt == Path("one/planner.md")
    assert config.prompts.verifier_prompt == Path("two/verifier.md")
    assert config.prompts.responder_prompt == Path("three/responder.md")
    assert config.metadata.tags == {"track": "native"}


def test_default_agent_config_uses_packaged_prompts(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_PROMPT_DIR", raising=False)
    config = ApiSettings.from_env().build_agent_config(AgentTrack.LANGGRAPH)

    assert config.prompts.planner_prompt.parent == DEFAULT_PROMPT_DIR
    manager = PromptManager(config.prompts)
    assert manager.planner().strip()
    assert manager.verifier().strip()
    assert manager.responder().strip()


def test_invalid_boolean_environment_value_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STREAM", "sometimes")
    with pytest.raises(ValueError, match="AGENT_STREAM"):
        ApiSettings.from_env()


def test_tauri_security_origins_and_hosts_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv(
        "API_CORS_ORIGINS",
        "tauri://localhost,http://localhost:5173",
    )
    monkeypatch.setenv("API_ALLOWED_HOSTS", "127.0.0.1,localhost")

    settings = ApiSettings.from_env()

    assert settings.cors_origins == (
        "tauri://localhost",
        "http://localhost:5173",
    )
    assert settings.allowed_hosts == ("127.0.0.1", "localhost")

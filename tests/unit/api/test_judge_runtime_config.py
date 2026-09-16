from __future__ import annotations

import pytest
from api.judge import (
    JUDGE_PROVIDERS,
    JudgeRuntimeConfig,
    ProviderCredentials,
    make_judge_factory,
)
from common.config import LLMConfig


def test_from_env_seeds_provider_and_credentials(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "groq")
    monkeypatch.setenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example")

    config = JudgeRuntimeConfig.from_env()

    assert config.provider == "groq"
    assert config.model == "llama-3.3-70b-versatile"
    assert config.credentials_for("groq").api_key == "env-key"
    assert config.credentials_for("openai").api_key == "oa-key"
    assert config.credentials_for("openai").base_url == "https://openai.example"


def test_from_env_judge_specific_overrides_carry_credentials(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://whatever.invalid")
    # A different provider's key must not leak into the judge selection.
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")

    config = JudgeRuntimeConfig.from_env()

    assert config.provider == "openai"
    assert config.credentials_for("openai").api_key == "judge-key"
    assert config.credentials_for("openai").base_url == "https://whatever.invalid"
    assert config.credentials_for("groq").api_key == "groq-key"


class TestResolve:
    def test_ollama_defaults_to_local_base(self) -> None:
        config = JudgeRuntimeConfig(provider="ollama", model="", credentials={})
        llm = config.resolve()
        assert isinstance(llm, LLMConfig)
        assert llm.provider == "ollama"
        assert llm.base_url == "http://localhost:11434"
        assert llm.model  # some provider default exists

    def test_ollama_normalizes_api_path(self) -> None:
        config = JudgeRuntimeConfig(
            provider="ollama",
            model="llama3.1",
            credentials={"ollama": ProviderCredentials(base_url="http://localhost:11434/api")},
        )
        assert config.resolve().base_url == "http://localhost:11434"

    def test_ollama_drops_own_api_key(self) -> None:
        # Ollama needs no key even if one was set in env by mistake.
        config = JudgeRuntimeConfig(
            provider="ollama", model="llama3.1", credentials={}
        )
        assert config.resolve().api_key == ""

    def test_cloud_provider_without_key_raises(self) -> None:
        config = JudgeRuntimeConfig(provider="groq", model="llama-3.3-70b-versatile", credentials={})
        with pytest.raises(ValueError, match="API key is required"):
            config.resolve()

    def test_per_run_override_beats_active_selection(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        config = JudgeRuntimeConfig(provider="groq", model="groq-default", credentials={})
        llm = config.resolve(provider="openai", model="gpt-4o-mini")
        assert llm.provider == "openai"
        assert llm.model == "gpt-4o-mini"
        assert llm.api_key == "oa-key"

    def test_ollama_url_dropped_for_cloud_provider(self) -> None:
        config = JudgeRuntimeConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            credentials={"groq": ProviderCredentials(base_url="http://localhost:11434", api_key="k")},
        )
        assert config.resolve().base_url is None

    def test_unknown_provider_rejected(self) -> None:
        config = JudgeRuntimeConfig(provider="ollama", model="", credentials={})
        with pytest.raises(ValueError, match="unsupported judge provider"):
            config.resolve(provider="together")

    def test_empty_selection_falls_back_to_default_model(self) -> None:
        config = JudgeRuntimeConfig(
            provider="groq",
            model="",
            credentials={"groq": ProviderCredentials(api_key="k")},
        )
        llm = config.resolve()
        assert bool(llm.model)
        assert llm.provider == "groq"


class TestUpdateAndPublic:
    def test_update_sets_fields_and_public_view_hides_secrets(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
        config = JudgeRuntimeConfig(provider="openai", model="gpt-4o-mini", credentials={})
        config.update(api_key="runtime-secret", base_url="https://openai.example")
        assert config.credentials_for("openai").api_key == "runtime-secret"
        public = config.public_settings()
        assert public["api_key_set"] is True
        assert "runtime-secret" not in str(public)
        assert "env-secret" not in str(public)

    def test_update_rejects_unknown_provider(self) -> None:
        config = JudgeRuntimeConfig(provider="ollama", model="", credentials={})
        with pytest.raises(ValueError, match="unsupported judge provider"):
            config.update(provider="together")

    def test_clear_api_key_override_falls_back_to_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
        config = JudgeRuntimeConfig(provider="openai", model="gpt-4o-mini", credentials={})
        config.update(api_key="runtime-secret")
        assert config._resolved_key("openai") == "runtime-secret"
        config.update(api_key="")
        assert config._resolved_key("openai") == "env-secret"

    def test_public_settings_lists_providers(self) -> None:
        config = JudgeRuntimeConfig(provider="ollama", model="", credentials={})
        public = config.public_settings()
        assert set(public["credentials"]) == set(JUDGE_PROVIDERS)


def test_make_judge_factory_builds_shared_contract_without_network() -> None:
    registry = JudgeRuntimeConfig(provider="ollama", model="llama3.1", credentials={})
    factory = make_judge_factory(registry)

    judge = factory("", "")
    assert hasattr(judge, "complete")
    assert judge.complete.usage_tracker is not None
    assert judge.model == "llama3.1"


def test_make_judge_factory_missing_agent_langgraph_is_a_validation_error(
    monkeypatch,
) -> None:
    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "agent_langgraph.runtime.judging",
        types.ModuleType("agent_langgraph.runtime.judging"),
    )
    registry = JudgeRuntimeConfig(provider="ollama", model="llama3.1", credentials={})
    factory = make_judge_factory(registry)

    with pytest.raises(ValueError, match="judge provider is unavailable"):
        factory("", "")
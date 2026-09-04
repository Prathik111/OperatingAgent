"""Tests for ``ModelProvider`` (``agent_langgraph.runtime.model_provider``).

The provider centralises model construction and dispatches on
``config.llm.provider``. The real ``langchain_*`` backends are not test
dependencies, so each is replaced with a fake module; the tests assert the
dispatch picks the right class and wires credentials/params correctly —
notably that the API key is wrapped in ``SecretStr`` and never passed raw.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_langgraph.runtime.model_provider import ModelProvider
from pydantic import SecretStr

from tests.support.langgraph import build_agent_config


class Recorder:
    """A fake chat-model class that records its construction kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def install_langchain(fake_module: Any, module_name: str, attr: str) -> None:
    fake_module(module_name, **{attr: Recorder})


def test_unknown_provider_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        ModelProvider(build_agent_config(provider="mystery"))


def test_openai_dispatch_wraps_secret(fake_module: Any) -> None:
    install_langchain(fake_module, "langchain_openai", "ChatOpenAI")
    provider = ModelProvider(build_agent_config(
        provider="openai", llm_timeout_seconds=12, llm_max_tokens=321
    ))
    model = provider.get_model()
    assert isinstance(model, Recorder)
    assert model.kwargs["model"] == "stub-model"
    assert isinstance(model.kwargs["api_key"], SecretStr)
    assert model.kwargs["api_key"].get_secret_value() == "test-key"
    assert model.kwargs["timeout"] == 12
    assert model.kwargs["max_completion_tokens"] == 321


def test_anthropic_dispatch_uses_model_name_and_timeout(fake_module: Any) -> None:
    install_langchain(fake_module, "langchain_anthropic", "ChatAnthropic")
    provider = ModelProvider(build_agent_config(
        provider="anthropic", llm_max_tokens=222, llm_top_p=0.8
    ))
    model = provider.get_model()
    assert model.kwargs["model_name"] == "stub-model"
    assert model.kwargs["timeout"] == 60
    assert isinstance(model.kwargs["api_key"], SecretStr)
    assert model.kwargs["stop"] is None
    assert model.kwargs["max_tokens_to_sample"] == 222
    assert model.kwargs["top_p"] == 0.8


def test_groq_dispatch(fake_module: Any) -> None:
    install_langchain(fake_module, "langchain_groq", "ChatGroq")
    provider = ModelProvider(build_agent_config(
        provider="groq", llm_timeout_seconds=13, llm_top_p=0.7
    ))
    model = provider.get_model()
    assert model.kwargs["model"] == "stub-model"
    assert isinstance(model.kwargs["api_key"], SecretStr)
    assert model.kwargs["timeout"] == 13
    assert model.kwargs["model_kwargs"] == {"top_p": 0.7}


def test_ollama_dispatch_needs_no_key(fake_module: Any) -> None:
    install_langchain(fake_module, "langchain_ollama", "ChatOllama")
    provider = ModelProvider(build_agent_config(
        provider="ollama", llm_timeout_seconds=14, llm_max_tokens=444
    ))
    model = provider.get_model()
    assert model.kwargs["model"] == "stub-model"
    assert "api_key" not in model.kwargs  # ollama is unauthenticated
    assert model.kwargs["num_predict"] == 444
    assert model.kwargs["client_kwargs"] == {"timeout": 14}


def test_get_model_returns_same_shared_instance(fake_module: Any) -> None:
    install_langchain(fake_module, "langchain_openai", "ChatOpenAI")
    provider = ModelProvider(build_agent_config(provider="openai"))
    assert provider.get_model() is provider.get_model()

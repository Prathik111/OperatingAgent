from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_langgraph.runtime.judging import (
    _text_of,
    _to_langchain_messages,
    _usage_of,
    judge_from_llm_config,
)
from common.config import LLMConfig


def test_to_langchain_messages_maps_roles() -> None:
    messages = [
        {"role": "system", "content": "You are the judge."},
        {"role": "user", "content": "Score this output."},
    ]
    mapped = _to_langchain_messages(messages)
    assert [type(item).__name__ for item in mapped] == [
        "SystemMessage",
        "HumanMessage",
    ]
    assert mapped[0].content == "You are the judge."
    assert mapped[1].content == "Score this output."


def test_to_langchain_messages_defaults_unknown_role_to_user() -> None:
    mapped = _to_langchain_messages([{"role": "assistant", "content": "hi"}])
    assert type(mapped[0]).__name__ == "HumanMessage"


def test_text_of_flattens_content_blocks() -> None:
    assert _text_of("plain") == "plain"
    assert _text_of(["a", {"type": "text", "text": "b"}, {"type": "image"}]) == "ab"
    assert _text_of({"unexpected": "object"}) == "{'unexpected': 'object'}"


def test_usage_of_computes_tokens_and_cost() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="k",
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    )
    result = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    tokens, cost = _usage_of(result, config)
    assert tokens == 150
    assert cost == pytest.approx((100 * 0.15 + 50 * 0.60) / 1_000_000)


def test_usage_of_tolerates_missing_metadata() -> None:
    config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="k")
    tokens, cost = _usage_of(SimpleNamespace(usage_metadata=None), config)
    assert tokens == 0
    assert cost == 0.0


def test_judge_from_llm_config_binds_model_and_tracker() -> None:
    judge = judge_from_llm_config(
        LLMConfig(provider="ollama", model="llama3.1", api_key="")
    )
    assert judge.model == "llama3.1"
    assert judge.complete.usage_tracker is not None
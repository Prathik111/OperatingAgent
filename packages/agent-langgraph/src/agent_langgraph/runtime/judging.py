"""LangChain-backed adapter for the provider-agnostic LLM judge.

The judge in :mod:`common.llm_judging` only knows a ``complete`` callable and a
model label. This module supplies that callable from any configured provider
(ollama, groq, openai, anthropic) by reusing :class:`ModelProvider`, so the
judge can run on a *different* provider than the agents it scores.
"""

from __future__ import annotations

from typing import Any

from common.config import LLMConfig
from common.llm_judging import JudgeUsageTracker, LLMJudge
from langchain_core.messages import HumanMessage, SystemMessage

from .model_provider import ModelProvider


def _to_langchain_messages(messages: list[dict[str, str]]) -> list[Any]:
    """Map the judge's ``{role, content}`` dicts onto LangChain messages."""
    mapped: list[Any] = []
    for message in messages:
        content = str(message.get("content", ""))
        if str(message.get("role", "user")) == "system":
            mapped.append(SystemMessage(content=content))
        else:
            mapped.append(HumanMessage(content=content))
    return mapped


def _text_of(content: Any) -> str:
    """Flatten a LangChain message ``content`` (str or content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _usage_of(result: Any, config: LLMConfig) -> tuple[int, float]:
    """Return ``(tokens, cost)`` from LangChain usage metadata, if reported."""
    usage = getattr(result, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cost = (
        input_tokens * config.input_price_per_million
        + output_tokens * config.output_price_per_million
    ) / 1_000_000
    return input_tokens + output_tokens, cost


def judge_from_llm_config(config: LLMConfig) -> LLMJudge:
    """Build one judge instance bound to ``config`` (a resolved judge provider)."""
    model = ModelProvider.create_chat_model(config)
    tracker = JudgeUsageTracker()

    async def complete(messages: list[dict[str, str]]) -> str:
        result = await model.ainvoke(_to_langchain_messages(messages))
        tokens, cost = _usage_of(result, config)
        if tokens or cost:
            tracker.add(tokens, cost)
        return _text_of(getattr(result, "content", result))

    complete.usage_tracker = tracker  # type: ignore[attr-defined]
    return LLMJudge(complete=complete, model=config.model)

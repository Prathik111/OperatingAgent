"""LLM clients: Groq primary, Ollama optional/local fallback (decision #7).

Both adapt to one normalized shape used by planner/executor/reflector:

    LLMResponse(text: str | None, tool_calls: list[ToolCall] | None, usage)

`ToolCall` is {id, name, arguments(dict)} - provider-specific JSON strings
are parsed here, so consumers never see raw provider shapes. Parsing failures
of the arguments string yield arguments={} + a marker in the call, never an
exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Protocol

from ..types import ToolInfo

SYSTEM_PROMPT = (
    "You are an operating agent that plans and executes multi-step goals on a "
    "local machine using sandboxed tools. Tool calls must use exactly the "
    "given tool names and JSON argument schemas. Never fabricate tool results. "
    "Keep outputs concise and factual."
)


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    parse_error: str | None = None


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tool_call(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolInfo] | None = None,
        *,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class GroqLLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str | None = None,
        max_retries: int = 2,
    ) -> None:
        from groq import AsyncGroq  # imported lazily so tests can run without the SDK

        client_kwargs: dict = {"api_key": api_key, "max_retries": max_retries, "timeout": 120.0}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncGroq(**client_kwargs)
        self.model = model
        self.max_retries = max_retries

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolInfo] | None = None,
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": 120.0,
        }
        if tools:
            kwargs["tools"] = [_to_provider_tool(t) for t in tools]
        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_calls = None
        if choice.tool_calls:
            tool_calls = [ToolCall(
                id=tc.id or "",
                name=tc.function.name or "",
                arguments=_parse_arguments(tc.function.arguments),
                parse_error=None if _parse_arguments_ok(tc.function.arguments) else "invalid json",
            ) for tc in choice.tool_calls]
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return LLMResponse(text=choice.content, tool_calls=tool_calls, usage=usage)


class OllamaLLMClient:
    """Local fallback via the ollama python client (optional extra)."""

    def __init__(self, model: str = "llama3.1:8b", host: str | None = None) -> None:
        import ollama  # lazy: optional dependency

        self.client = ollama.AsyncClient(host=host) if host else ollama.AsyncClient()
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolInfo] | None = None,
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict = {"model": self.model, "messages": messages, "options": {"temperature": temperature}}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.schema.input_schema,
            }} for t in tools]
        resp = await self.client.chat(**kwargs)
        msg = resp.get("message") or {}
        tool_calls = None
        raw_calls = msg.get("tool_calls") or []
        if raw_calls:
            tool_calls = []
            for tc in raw_calls:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments") or {}
                tool_calls.append(ToolCall(
                    id=f"ollama-{len(tool_calls)}",
                    name=fn.get("name", ""),
                    arguments=args_raw if isinstance(args_raw, dict) else _parse_arguments(args_raw),
                ))
        usage = Usage(
            input_tokens=(resp.get("prompt_eval_count") or 0),
            output_tokens=(resp.get("eval_count") or 0),
        )
        return LLMResponse(text=msg.get("content"), tool_calls=tool_calls, usage=usage)


def build_llm(settings) -> LLMClient:
    """Wire the configured provider (decision #7: groq default)."""
    if settings.llm_provider == "ollama":
        return OllamaLLMClient(model=settings.ollama_model)
    api_key = settings.groq_api_key
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set (provider=groq). Set it, use provider=ollama, "
            "or set AGENT_NATIVE_LLM_PROVIDER."
        )
    return GroqLLMClient(api_key=api_key, model=settings.groq_model, base_url=settings.groq_base_url)


def _to_provider_tool(t: ToolInfo) -> dict:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.schema.input_schema,
        },
    }


def _parse_arguments(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_arguments_ok(raw: str | None) -> bool:
    if not raw:
        return True
    try:
        parsed = json.loads(raw)
        return isinstance(parsed, dict)
    except (json.JSONDecodeError, TypeError):
        return False

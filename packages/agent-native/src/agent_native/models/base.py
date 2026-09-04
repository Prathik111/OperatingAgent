"""Small contracts shared by native model providers.

The agent loop deliberately depends on this module rather than on a provider SDK.
Provider adapters translate SDK-specific chunks into :class:`StreamEvent` values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class StreamType(str, Enum):
    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    USAGE = "usage"
    DONE = "done"


class ToolFormat(str, Enum):
    NATIVE = "native"
    JSON = "json"
    TEXT = "text"


@dataclass(frozen=True)
class StreamEvent:
    type: StreamType
    data: dict[str, Any]


@dataclass(frozen=True)
class Model:
    provider: str
    model_id: str
    context_size: int = 128_000
    max_output: int = 8_192
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    tool_format: ToolFormat = ToolFormat.NATIVE
    cache_marker: str = ""
    supports_vision: bool = False

    def cost_of(self, input_tokens: int, output_tokens: int) -> float:
        """Return the provider-reported token cost in USD."""
        return (
            max(0, input_tokens) * self.input_price_per_million
            + max(0, output_tokens) * self.output_price_per_million
        ) / 1_000_000


class ModelRegistry:
    """Registry joining configured model names to provider instances."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._models: dict[str, Model] = {}

    def register_provider(self, name: str, provider: Any) -> None:
        if not name:
            raise ValueError("provider name cannot be empty")
        self._providers[name] = provider

    def register_model(self, name: str, model: Model) -> None:
        if not name:
            raise ValueError("model name cannot be empty")
        if not isinstance(model, Model):
            raise TypeError("model must be a Model")
        self._models[name] = model

    def get(self, name: str) -> Model:
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {name}") from exc

    def get_provider(self, model: Model | str) -> Any:
        resolved = self.get(model) if isinstance(model, str) else model
        try:
            return self._providers[resolved.provider]
        except KeyError as exc:
            raise KeyError(f"No provider registered for: {resolved.provider}") from exc

    def list_models(self) -> list[Model]:
        return list(self._models.values())

    def list_model_names(self) -> list[str]:
        return list(self._models)


def rough_token_count(value: Any) -> int:
    """Estimate tokens for budgeting when a provider gives no tokenizer."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return max(1, len(text) // 4) if text else 0


def stable_prefix_fingerprint(messages: list, tools: list) -> str:
    """Hash the leading system prompt and tool declarations, ignoring the tail."""
    prefix: list = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        prefix.append(message)
    payload = json.dumps([prefix, tools], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mark_cacheable_prefix(messages: list, marker: str) -> list:
    """Mark the last leading system message for providers supporting prompt caching."""
    if not marker:
        return messages
    last_system = -1
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        last_system = index
    if last_system < 0:
        return messages
    marked = list(messages)
    entry = dict(marked[last_system])
    entry[marker] = {"type": "ephemeral"}
    marked[last_system] = entry
    return marked


def wire_has_media(messages: list) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("images"):
            return True
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


def require_vision_support(messages: list, model: Model) -> None:
    if wire_has_media(messages) and not model.supports_vision:
        raise RuntimeError(f"Model {model.model_id!r} does not support vision input")

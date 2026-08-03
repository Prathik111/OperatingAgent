from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolSchema:

    input_schema: dict[str, Any]

    output_schema: dict[str, Any]


@dataclass(slots=True)
class ToolInfo:

    name: str

    description: str

    schema: ToolSchema

    risk_level: str = "safe"


@dataclass(slots=True)
class ToolCallRequest:

    tool_name: str

    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolCallResult:

    success: bool

    output: Any

    error: str | None = None
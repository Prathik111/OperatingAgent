"""Shared fakes: scripted LLM, scripted MCP client, record-events sink."""

from __future__ import annotations

from collections import deque

import pytest

from agent_native.events import AgentEvent
from agent_native.llm import LLMResponse, ToolCall, Usage
from agent_native.mcp import MCPClient
from agent_native.repository import InMemoryTaskRepository
from agent_native.types import ToolCallResult, ToolInfo


class FakeLLM:
    """Scripted responses; when the script is exhausted the last one repeats."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = deque(script)
        self.calls: list[dict] = []
        self.usage = Usage(input_tokens=10, output_tokens=5)

    async def complete(self, messages, tools=None, *, temperature=0.0):
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if self._script:
            response = self._script.popleft()
            self._script.append(response)
            return response
        return LLMResponse(text="done", usage=self.usage)

    @classmethod
    def text(cls, text: str) -> LLMResponse:
        return LLMResponse(text=text, usage=Usage(input_tokens=10, output_tokens=5))

    @classmethod
    def tool(cls, name: str, arguments: dict, call_id: str = "c1") -> LLMResponse:
        return LLMResponse(
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
            usage=Usage(input_tokens=10, output_tokens=5),
        )


class FakeMCP(MCPClient):
    """Scripted tool results; unknown tools fail."""

    def __init__(self, results: dict[str, ToolCallResult] | None = None) -> None:
        self.results = dict(results or {})
        self.calls: list[tuple[str, dict]] = []
        self.tools: list[ToolInfo] = []
        self.closed = False

    def add_tool(self, tool: ToolInfo) -> None:
        self.tools.append(tool)

    async def list_tools(self) -> list[ToolInfo]:
        return list(self.tools)

    async def call_tool(self, request):
        self.calls.append((request.tool_name, request.arguments))
        if request.tool_name in self.results:
            return self.results[request.tool_name]
        return ToolCallResult(success=False, output=None, error=f"unknown tool {request.tool_name!r}")

    async def close(self) -> None:
        self.closed = True


class EventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


def make_tool(name: str, risk_level: str = "safe") -> ToolInfo:
    from agent_native.types import ToolSchema

    return ToolInfo(
        name=name,
        description=f"tool {name}",
        schema=ToolSchema(input_schema={"type": "object"}, output_schema={}),
        risk_level=risk_level,
    )


@pytest.fixture
def memory_repo():
    return InMemoryTaskRepository()


@pytest.fixture
def sink() -> EventSink:
    return EventSink()


@pytest.fixture
def executor_tools():
    from agent_native.types import ToolSchema

    def _tool(name: str) -> ToolInfo:
        return ToolInfo(
            name=name, description=f"tool {name}",
            schema=ToolSchema(input_schema={"type": "object", "properties": {}}, output_schema={}),
        )

    return [_tool("read_file"), _tool("write_file"), _tool("run_command")]
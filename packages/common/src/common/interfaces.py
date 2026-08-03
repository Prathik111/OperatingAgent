"""IMCPClient and IAgentOrchestrator — the two interfaces that let both
tracks be swapped behind TaskService without it knowing which one it holds.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from .agent import AgentRunResult, AgentTask
from .events import AgentEvent
from .tools import ToolCallRequest, ToolCallResult, ToolInfo

EventCallback = Callable[[AgentEvent], Awaitable[None] | None]

class IMCPClient(Protocol):

    async def list_tools(self) -> list[ToolInfo]: ...

    async def call_tool(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResult:
        ...


class IAgentOrchestrator(Protocol):

    async def run( self, task: AgentTask, on_event: EventCallback | None = None,
    ) -> AgentRunResult: ...
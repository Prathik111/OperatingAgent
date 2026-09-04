"""The production FastMCP gateway launched through its stdio transport."""

from __future__ import annotations

import sys

from agent_langgraph.mcp_adapter import MCPAdapter
from common.tools import ToolCallRequest


async def test_gateway_subprocess_lists_and_calls_tools() -> None:
    adapter = MCPAdapter.from_stdio(sys.executable, ["-m", "gateway_server"])
    try:
        tools = await adapter.list_tools()
        names = {tool.name for tool in tools}
        assert "gateway_health" in names
        assert any(name.startswith("filesystem_") for name in names)

        result = await adapter.call_tool(
            ToolCallRequest(tool_name="gateway_health", arguments={})
        )
        assert result.success is True, result.error
        assert "healthy" in str(result.output)
    finally:
        await adapter.aclose()

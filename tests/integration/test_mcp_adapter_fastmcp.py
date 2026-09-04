"""Integration: ``MCPAdapter`` + ``ToolRegistry`` over the real MCP protocol.

``fastmcp.Client`` accepts a ``FastMCP`` instance directly, so the genuine
protocol runs in-process — the adapter's per-call ``async with self._client``
session handling, result mapping, and error translation are all exercised
without a gateway process or a network hop.

This is the seam that unit tests deliberately fake (``FakeAdapter``,
``StubToolRegistry``), so it is the layer that proves those fakes are honest.
"""

from __future__ import annotations

import pytest
from agent_langgraph.mcp_adapter import MCPAdapter
from agent_langgraph.runtime.tool_registry import ToolRegistry
from common.tools import ToolCallRequest


@pytest.fixture
def in_process_server():
    """A tiny FastMCP server exposing a success tool and a failing tool."""
    fastmcp = pytest.importorskip("fastmcp")

    server = fastmcp.FastMCP(name="test-server")

    @server.tool
    def echo(text: str) -> dict:
        """Echo the provided text."""
        return {"echoed": text}

    @server.tool
    def boom() -> dict:
        """Always fails."""
        raise ValueError("kaboom")

    return server


async def test_list_tools_maps_server_tools(in_process_server) -> None:
    adapter = MCPAdapter(in_process_server)
    tools = await adapter.list_tools()
    names = {t.name for t in tools}
    assert {"echo", "boom"} <= names
    echo = next(t for t in tools if t.name == "echo")
    assert "text" in echo.schema.input_schema.get("properties", {})


async def test_call_tool_success(in_process_server) -> None:
    adapter = MCPAdapter(in_process_server)
    result = await adapter.call_tool(ToolCallRequest(tool_name="echo", arguments={"text": "hi"}))
    assert result.success is True
    assert result.error is None
    assert "hi" in str(result.output)


@pytest.mark.regression
async def test_call_tool_business_error_maps_to_failure(in_process_server) -> None:
    """A tool that raises must come back as a failed result, not an exception —
    the executor's retry logic depends on being handed a result object."""
    adapter = MCPAdapter(in_process_server)
    result = await adapter.call_tool(ToolCallRequest(tool_name="boom", arguments={}))
    assert result.success is False
    assert result.output is None
    assert result.error  # a non-empty message


@pytest.mark.regression
async def test_call_unknown_tool_is_failure_not_exception(in_process_server) -> None:
    """A protocol-level failure (unknown tool) is caught and returned as a
    failed ToolCallResult rather than raising."""
    adapter = MCPAdapter(in_process_server)
    result = await adapter.call_tool(ToolCallRequest(tool_name="does_not_exist", arguments={}))
    assert result.success is False
    assert result.error


# ---------------------------------------------------------------------------
# Through the registry the executor actually uses
# ---------------------------------------------------------------------------


async def test_registry_over_real_adapter_round_trips(in_process_server) -> None:
    """``call_by_name`` is what ExecutorNode calls; prove it works against the
    real adapter, not just the ``FakeAdapter`` the unit tests use."""
    registry = ToolRegistry(MCPAdapter(in_process_server))

    tools = await registry.list_tools()
    assert "echo" in {t.name for t in tools}

    result = await registry.call_by_name("echo", {"text": "round trip"})
    assert result.success is True
    assert "round trip" in str(result.output)

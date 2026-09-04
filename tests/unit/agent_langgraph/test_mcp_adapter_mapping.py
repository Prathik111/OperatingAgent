"""Unit tests for ``MCPAdapter``'s pure wire-type mapping.

The adapter is the LangGraph track's ``IMCPClient``: it maps FastMCP wire types
onto the project's ``ToolInfo`` / ``ToolCallResult`` so nothing downstream
depends on FastMCP. These are the pure, side-effect-free helpers; the adapter
driven over the real protocol lives in
``tests/integration/test_mcp_adapter_fastmcp.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from agent_langgraph.mcp_adapter import MCPAdapter
from fastmcp.client.transports import StdioTransport


def test_to_tool_info_maps_fields() -> None:
    tool = SimpleNamespace(
        name="echo",
        description="echoes text",
        inputSchema={"type": "object"},
        outputSchema={"type": "object"},
        annotations=None,
    )
    info = MCPAdapter._to_tool_info(tool)
    assert info.name == "echo"
    assert info.description == "echoes text"
    assert info.schema.input_schema == {"type": "object"}
    assert info.risk_level == "safe"  # default when no annotation


def test_to_tool_info_reads_risk_annotation() -> None:
    tool = SimpleNamespace(
        name="rm", description=None, inputSchema=None, outputSchema=None,
        annotations=SimpleNamespace(risk_level="blocked"),
    )
    info = MCPAdapter._to_tool_info(tool)
    assert info.risk_level == "blocked"
    assert info.description == ""  # None description normalised to empty string
    assert info.schema.input_schema == {}


def test_error_text_extracts_message() -> None:
    result = SimpleNamespace(content=[SimpleNamespace(text="boom happened")])
    assert MCPAdapter._error_text(result) == "boom happened"


def test_error_text_defaults_when_no_content() -> None:
    assert MCPAdapter._error_text(SimpleNamespace(content=None)) == "tool reported an error"


def test_from_stdio_builds_explicit_subprocess_transport() -> None:
    adapter = MCPAdapter.from_stdio(sys.executable, ["-m", "gateway_server"])
    transport = adapter._client.transport

    assert isinstance(transport, StdioTransport)
    assert transport.command == sys.executable
    assert transport.args == ["-m", "gateway_server"]

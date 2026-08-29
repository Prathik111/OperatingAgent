"""Tools: the contract, the dispatcher, the MCP bridge."""

from __future__ import annotations

from .base import (
    ArgumentChecker,
    ExecutionMode,
    Tool,
    ToolDefinition,
    ToolPermissions,
    ToolRegistry,
    ToolResult,
    native_schema,
)
from .builtins import default_tools
from .manager import ToolManager
from .mcp_bridge import MCPTool, MCPToolProvider

__all__ = [
    "ArgumentChecker",
    "ExecutionMode",
    "Tool",
    "ToolDefinition",
    "ToolPermissions",
    "ToolRegistry",
    "ToolResult",
    "native_schema",
    "ToolManager",
    "MCPTool",
    "MCPToolProvider",
    "default_tools",
]

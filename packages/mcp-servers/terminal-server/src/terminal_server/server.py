"""
Terminal MCP Server using FastMCP.

This module creates the FastMCP server and registers all terminal tools.
Business logic remains inside TerminalService.
"""

from __future__ import annotations

from typing import Final

from fastmcp import FastMCP

from .services.terminal_service import TerminalService
from .tools.list_processes import register_list_processes
from .tools.run_command import register_run_command

VERSION: Final[str] = "0.1.0"

mcp = FastMCP(
    name="terminal-server",
    version=VERSION,
)

terminal_service = TerminalService()

register_run_command(mcp, terminal_service)
register_list_processes(mcp, terminal_service)


@mcp.tool
def health() -> dict:
    """Health endpoint."""
    return {
        "status": "healthy",
        "server": "terminal-server",
        "version": VERSION,
    }


if __name__ == "__main__":
    mcp.run()

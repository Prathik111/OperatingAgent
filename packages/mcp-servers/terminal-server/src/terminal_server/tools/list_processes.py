"""
FastMCP tool for listing running processes.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.terminal_service import TerminalService


def register_list_processes(
    mcp: FastMCP,
    service: TerminalService,
) -> None:
    """Register the list_processes MCP tool."""

    @mcp.tool(
        name="list_processes",
        description="List the currently running processes.",
    )
    def list_processes():
        """
        List processes.

        Returns:
            Result from the terminal service.
        """

        return service.list_processes()

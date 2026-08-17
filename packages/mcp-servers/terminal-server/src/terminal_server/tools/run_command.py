"""
FastMCP tool for running shell commands.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.terminal_service import TerminalService


def register_run_command(
    mcp: FastMCP,
    service: TerminalService,
) -> None:
    """Register the run_command MCP tool."""

    @mcp.tool(
        name="run_command",
        description="Run a shell command and return its output.",
    )
    def run_command(
        command: str,
        cwd: str | None = None,
        timeout: float = 60.0,
    ):
        """
        Run a command.

        Args:
            command: Shell command to run.
            cwd: Working directory for the command.
            timeout: Maximum execution time in seconds.

        Returns:
            Result from the terminal service.
        """

        if not command.strip():
            raise ValueError("command must be provided.")

        return service.run_command(
            command=command,
            cwd=cwd,
            timeout=timeout,
        )

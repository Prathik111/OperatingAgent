"""
Terminal MCP Server using FastMCP.

This module creates the FastMCP server and registers all terminal tools.
Business logic remains inside TerminalService.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from fastmcp import FastMCP

from .services.terminal_service import TerminalService
from .tools.list_processes import register_list_processes
from .tools.run_command import register_run_command

VERSION: Final[str] = "0.1.0"

#: Set to a falsey value ("0", "false", "no") to drop ``run_command`` entirely.
#: Do this whenever the server - or a gateway mounting it - is exposed over a
#: network transport to clients you do not fully trust. ``list_processes``
#: stays registered either way.
RUN_COMMAND_ENV_VAR: Final[str] = "TERMINAL_SERVER_ENABLE_RUN_COMMAND"

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def _run_command_enabled() -> bool:
    """Whether the command-execution tool should be exposed at all."""

    return os.environ.get(RUN_COMMAND_ENV_VAR, "1").strip().lower() not in {"0", "false", "no"}


def build_terminal_server(root: str | None = None) -> FastMCP:
    """Build a terminal server confined to ``root`` when supplied."""
    server = FastMCP(
        name="terminal-server",
        version=VERSION,
        mask_error_details=True,
    )
    service = TerminalService(root=root)
    if _run_command_enabled():
        register_run_command(server, service)
    register_list_processes(server, service)

    @server.tool
    def health() -> dict:
        return {"status": "healthy", "server": "terminal-server", "version": VERSION}

    server._operating_agent_terminal_service = service  # pyright: ignore[reportAttributeAccessIssue]
    return server


mcp = build_terminal_server()
terminal_service = mcp._operating_agent_terminal_service  # pyright: ignore[reportAttributeAccessIssue]


if _run_command_enabled():
    LOGGER.info(
        "run_command registered with allowlist=%s",
        ", ".join(sorted(terminal_service.allowed_commands)),
    )
else:
    LOGGER.warning("run_command disabled via %s", RUN_COMMAND_ENV_VAR)

if __name__ == "__main__":
    mcp.run()

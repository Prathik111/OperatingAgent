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

mcp = FastMCP(
    name="terminal-server",
    version=VERSION,
)

terminal_service = TerminalService()


def _run_command_enabled() -> bool:
    """Whether the command-execution tool should be exposed at all."""

    return os.environ.get(RUN_COMMAND_ENV_VAR, "1").strip().lower() not in {"0", "false", "no"}


if _run_command_enabled():
    register_run_command(mcp, terminal_service)
    LOGGER.info(
        "run_command registered with allowlist=%s",
        ", ".join(sorted(terminal_service.allowed_commands)),
    )
else:
    LOGGER.warning("run_command disabled via %s", RUN_COMMAND_ENV_VAR)

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

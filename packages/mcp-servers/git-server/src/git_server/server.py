"""
Git MCP Server using FastMCP.

This module creates the FastMCP server and registers all git tools.
Business logic remains inside GitService.
"""

from __future__ import annotations

from typing import Final

from fastmcp import FastMCP

from .services.git_service import GitService
from .tools.diff import register_diff
from .tools.git_log import register_git_log
from .tools.git_status import register_git_status
from .tools.list_branches import register_list_branches

VERSION: Final[str] = "0.1.0"

mcp = FastMCP(
    name="git-server",
    version=VERSION,
)

git_service = GitService()

register_git_status(mcp, git_service)
register_list_branches(mcp, git_service)
register_git_log(mcp, git_service)
register_diff(mcp, git_service)


@mcp.tool
def health() -> dict:
    """Health endpoint."""
    return {
        "status": "healthy",
        "server": "git-server",
        "version": VERSION,
    }


if __name__ == "__main__":
    mcp.run()

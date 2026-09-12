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

def build_git_server(root: str | None = None) -> FastMCP:
    """Build a Git server confined to ``root`` when supplied."""
    server = FastMCP(name="git-server", version=VERSION, mask_error_details=True)
    service = GitService(root=root)
    register_git_status(server, service)
    register_list_branches(server, service)
    register_git_log(server, service)
    register_diff(server, service)

    @server.tool
    def health() -> dict:
        return {"status": "healthy", "server": "git-server", "version": VERSION}

    server._operating_agent_git_service = service  # pyright: ignore[reportAttributeAccessIssue]
    return server


mcp = build_git_server()
git_service = mcp._operating_agent_git_service  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    mcp.run()

"""
FastMCP tool for reading recent Git log entries.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.git_service import GitService


def register_git_log(
    mcp: FastMCP,
    service: GitService,
) -> None:
    """Register the git_log MCP tool."""

    @mcp.tool(
        name="git_log",
        description="Return a short Git history log for the target repository.",
    )
    def git_log(
        repository: str = ".",
        max_count: int = 10,
    ):
        """
        Show git log.

        Args:
            repository: Path to the git repository.
            max_count: Maximum number of commits to return.

        Returns:
            Result from the git service.
        """

        if not repository.strip():
            raise ValueError("repository must be provided.")

        return service.log(
            repository=repository,
            max_count=max_count,
        )

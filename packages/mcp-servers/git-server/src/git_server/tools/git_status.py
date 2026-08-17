"""
FastMCP tool for reading git working-tree status.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.git_service import GitService


def register_git_status(
    mcp: FastMCP,
    service: GitService,
) -> None:
    """Register the git_status MCP tool."""

    @mcp.tool(
        name="git_status",
        description="Return the Git working tree status for a repository.",
    )
    def git_status(
        repository: str = ".",
    ):
        """
        Show git status.

        Args:
            repository: Path to the git repository.

        Returns:
            Result from the git service.
        """

        if not repository.strip():
            raise ValueError("repository must be provided.")

        return service.status(
            repository=repository,
        )

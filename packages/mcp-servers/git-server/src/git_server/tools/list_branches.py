"""
FastMCP tool for listing Git branches.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.git_service import GitService


def register_list_branches(
    mcp: FastMCP,
    service: GitService,
) -> None:
    """Register the list_branches MCP tool."""

    @mcp.tool(
        name="list_branches",
        description="List Git branches for the target repository.",
    )
    def list_branches(
        repository: str = ".",
    ):
        """
        List branches.

        Args:
            repository: Path to the git repository.

        Returns:
            Result from the git service.
        """

        if not repository.strip():
            raise ValueError("repository must be provided.")

        return service.branches(
            repository=repository,
        )

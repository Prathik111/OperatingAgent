"""
FastMCP tool for fetching Git diffs.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.git_service import GitService


def register_diff(
    mcp: FastMCP,
    service: GitService,
) -> None:
    """Register the diff MCP tool."""

    @mcp.tool(
        name="diff",
        description="Return a Git diff payload for the target repository and revision.",
    )
    def diff(
        repository: str = ".",
        target: str = "HEAD",
    ):
        """
        Show git diff.

        Args:
            repository: Path to the git repository.
            target: Target revision to diff against.

        Returns:
            Result from the git service.
        """

        if not repository.strip():
            raise ValueError("repository must be provided.")

        return service.diff(
            repository=repository,
            target=target,
        )

"""
FastMCP tool for checking path existence.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_exists(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the exists MCP tool."""

    @mcp.tool(
        name="exists",
        description="Check whether a path exists and whether it is a file or directory.",
    )
    def exists(
        path: str,
    ):
        """
        Check a path.

        Args:
            path: Path to check.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return service.exists(
            path=path,
        )

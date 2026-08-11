"""
FastMCP tool for path metadata collection.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_metadata(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the metadata MCP tool."""

    @mcp.tool(
        name="metadata",
        description="Collect metadata for a file or directory path.",
    )
    def metadata(
        path: str,
    ):
        """
        Collect metadata for a path.

        Args:
            path: File or directory path.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return service.metadata(
            path=path,
        )

"""
FastMCP tool for renaming files.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_rename_file(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the rename_file MCP tool."""

    @mcp.tool(
        name="rename_file",
        description="Rename a file to a new file path.",
    )
    def rename_file(
        source: str,
        destination: str,
    ):
        """
        Rename a file.

        Args:
            source: Source file path.
            destination: New file path.

        Returns:
            Result from the filesystem service.
        """

        if not source.strip():
            raise ValueError("source must be provided.")

        if not destination.strip():
            raise ValueError("destination must be provided.")

        return service.rename_file(
            source=source,
            destination=destination,
        )

"""
FastMCP tool for moving files.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_move_file(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the move_file MCP tool."""

    @mcp.tool(
        name="move_file",
        description="Move a file to a new destination path.",
    )
    def move_file(
        source: str,
        destination: str,
        overwrite: bool = True,
    ):
        """
        Move a file.

        Args:
            source: Source file path.
            destination: Destination file path.
            overwrite: Whether to overwrite the destination file.

        Returns:
            Result from the filesystem service.
        """

        if not source.strip():
            raise ValueError("source must be provided.")

        if not destination.strip():
            raise ValueError("destination must be provided.")

        return service.move_file(
            source=source,
            destination=destination,
            overwrite=overwrite,
        )

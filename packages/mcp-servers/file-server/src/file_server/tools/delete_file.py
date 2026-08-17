"""
FastMCP tool for deleting files.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_delete_file(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the delete_file MCP tool."""

    @mcp.tool(
        name="delete_file",
        description="Delete a file from the local filesystem.",
    )
    def delete_file(
        path: str,
    ):
        """
        Delete a file.

        Args:
            path: File path to delete.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return service.delete_file(
            path=path,
        )

"""
FastMCP tool for deleting directories.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_delete_directory(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the delete_directory MCP tool."""

    @mcp.tool(
        name="delete_directory",
        description="Delete a directory from the local filesystem.",
    )
    def delete_directory(
        path: str,
        recursive: bool = False,
    ):
        """
        Delete a directory.

        Args:
            path: Directory path to delete.
            recursive: Whether to delete the directory tree recursively.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return service.delete_directory(
            path=path,
            recursive=recursive,
        )

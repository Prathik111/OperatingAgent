"""
FastMCP tool for listing directory contents.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_list_directory(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the list_directory MCP tool."""

    @mcp.tool(
        name="list_directory",
        description="List the contents of a directory path.",
    )
    def list_directory(
        path: str,
        recursive: bool = False,
    ):
        """
        List a directory.

        Args:
            path: Directory path to list.
            recursive: Whether to recurse into child directories.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return service.list_directory(
            path=path,
            recursive=recursive,
        )

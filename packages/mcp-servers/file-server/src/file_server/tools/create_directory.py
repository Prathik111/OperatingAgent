"""
FastMCP tool for creating directories.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_create_directory(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the create_directory MCP tool."""

    @mcp.tool(
        name="create_directory",
        description="Create a directory path on the local filesystem.",
    )
    def create_directory(
        path: str,
        parents: bool = True,
    ):
        """
        Create a directory.

        Args:
            path: Directory path to create.
            parents: Whether to create missing parent directories.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return service.create_directory(
            path=path,
            parents=parents,
        )

"""
FastMCP tool for searching file names inside a directory tree.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_search_files(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the search_files MCP tool."""

    @mcp.tool(
        name="search_files",
        description="Search the file system for file names matching a query string.",
    )
    def search_files(
        root: str,
        query: str,
        recursive: bool = True,
    ):
        """
        Search for files.

        Args:
            root: Root directory to search.
            query: File name query term.
            recursive: Whether to recurse into child directories.

        Returns:
            Result from the filesystem service.
        """

        if not root.strip():
            raise ValueError("root must be provided.")

        if not query.strip():
            raise ValueError("query must be provided.")

        return service.search_files(
            root=root,
            query=query,
            recursive=recursive,
        )

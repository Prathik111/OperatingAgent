"""
FastMCP tool for reading files.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_read_file(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the read_file MCP tool."""

    @mcp.tool(
        name="read_file",
        description="Read a text file from the local filesystem.",
    )
    def read_file(
        path: str,
        encoding: str = "utf-8",
    ):
        """
        Read a file.

        Args:
            path: File path to read.
            encoding: Text encoding used to decode the file.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return service.read_file(
            path=path,
            encoding=encoding,
        )

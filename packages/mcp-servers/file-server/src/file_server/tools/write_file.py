"""
FastMCP tool for writing files.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_write_file(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the write_file MCP tool."""

    @mcp.tool(
        name="write_file",
        description="Write text content to a file path on the local filesystem.",
    )
    def write_file(
        path: str,
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = True,
    ):
        """
        Write a file.

        Args:
            path: File path to write.
            content: Text content to write.
            encoding: Text encoding used to encode the file.
            overwrite: Whether to overwrite the file if it exists.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        if content is None:
            raise ValueError("content must be provided.")

        return service.write_file(
            path=path,
            content=content,
            encoding=encoding,
            overwrite=overwrite,
        )

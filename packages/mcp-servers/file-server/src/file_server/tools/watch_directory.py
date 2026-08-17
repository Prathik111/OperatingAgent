"""
FastMCP tool for watching a directory over a bounded interval.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_watch_directory(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the watch_directory MCP tool."""

    @mcp.tool(
        name="watch_directory",
        description="Collect a short sequence of directory snapshots over time.",
    )
    async def watch_directory(
        path: str,
        interval_seconds: float = 1.0,
        limit: int = 1,
    ):
        """
        Watch a directory.

        Args:
            path: Directory path to watch.
            interval_seconds: Seconds between snapshots.
            limit: Number of snapshots to collect.

        Returns:
            Result from the filesystem service.
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        return await service.watch_directory(
            path=path,
            interval_seconds=interval_seconds,
            limit=limit,
        )

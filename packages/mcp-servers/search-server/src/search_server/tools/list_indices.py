"""
FastMCP tool for listing search indices.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.search_service import SearchService


def register_list_indices(
    mcp: FastMCP,
    service: SearchService,
) -> None:
    """Register the list_indices MCP tool."""

    @mcp.tool(
        name="list_indices",
        description="List the currently available in-memory search indices.",
    )
    def list_indices():
        """
        List indices.

        Returns:
            Result from the search service.
        """

        return service.list_indices()

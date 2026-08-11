"""
FastMCP tool for searching indexed documents.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.search_service import SearchService


def register_search_documents(
    mcp: FastMCP,
    service: SearchService,
) -> None:
    """Register the search_documents MCP tool."""

    @mcp.tool(
        name="search_documents",
        description="Search a named index for documents that match a query term.",
    )
    def search_documents(
        index_name: str,
        query: str,
    ):
        """
        Search documents.

        Args:
            index_name: Name of the index to search.
            query: Query term to match against document content.

        Returns:
            Result from the search service.
        """

        if not index_name.strip():
            raise ValueError("index_name must be provided.")

        if not query.strip():
            raise ValueError("query must be provided.")

        return service.search(
            index_name=index_name,
            query=query,
        )

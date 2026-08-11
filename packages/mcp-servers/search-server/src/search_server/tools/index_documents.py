"""
FastMCP tool for indexing search documents.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..services.search_service import SearchService


def register_index_documents(
    mcp: FastMCP,
    service: SearchService,
) -> None:
    """Register the index_documents MCP tool."""

    @mcp.tool(
        name="index_documents",
        description="Create or replace an in-memory document index.",
    )
    def index_documents(
        index_name: str,
        documents: list[dict[str, Any]],
    ):
        """
        Index documents.

        Args:
            index_name: Name of the index to store documents under.
            documents: Documents to store in the index.

        Returns:
            Result from the search service.
        """

        if not index_name.strip():
            raise ValueError("index_name must be provided.")

        return service.index_documents(
            index_name=index_name,
            documents=documents,
        )

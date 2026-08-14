"""
Search MCP Server using FastMCP.

This module creates the FastMCP server and registers all search tools.
Business logic remains inside SearchService.
"""

from __future__ import annotations

from typing import Final

from fastmcp import FastMCP

from .services.search_service import SearchService
from .tools.index_documents import register_index_documents
from .tools.list_indices import register_list_indices
from .tools.search_documents import register_search_documents

VERSION: Final[str] = "0.1.0"

mcp = FastMCP(
    name="search-server",
    version=VERSION,
    mask_error_details=True,
)

search_service = SearchService()

register_index_documents(mcp, search_service)
register_search_documents(mcp, search_service)
register_list_indices(mcp, search_service)


@mcp.tool
def health() -> dict:
    """Health endpoint."""
    return {
        "status": "healthy",
        "server": "search-server",
        "version": VERSION,
    }


if __name__ == "__main__":
    mcp.run()

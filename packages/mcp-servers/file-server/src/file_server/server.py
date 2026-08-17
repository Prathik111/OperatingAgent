"""
Filesystem MCP Server using FastMCP.

This module creates the FastMCP server and registers all filesystem tools.
Business logic remains inside FileSystemService.
"""

from __future__ import annotations

from typing import Final

from fastmcp import FastMCP

from .services.filesystem_service import FileSystemService
from .tools.copy_file import register_copy_file
from .tools.create_directory import register_create_directory
from .tools.delete_directory import register_delete_directory
from .tools.delete_file import register_delete_file
from .tools.exists import register_exists
from .tools.list_directory import register_list_directory
from .tools.metadata import register_metadata
from .tools.move_file import register_move_file
from .tools.read_file import register_read_file
from .tools.rename_file import register_rename_file
from .tools.search_files import register_search_files
from .tools.watch_directory import register_watch_directory
from .tools.write_file import register_write_file

VERSION: Final[str] = "0.1.0"

mcp = FastMCP(
    name="file-server",
    version=VERSION,
    mask_error_details=True,
)

filesystem_service = FileSystemService()

register_read_file(mcp, filesystem_service)
register_write_file(mcp, filesystem_service)
register_delete_file(mcp, filesystem_service)
register_copy_file(mcp, filesystem_service)
register_move_file(mcp, filesystem_service)
register_rename_file(mcp, filesystem_service)
register_list_directory(mcp, filesystem_service)
register_create_directory(mcp, filesystem_service)
register_delete_directory(mcp, filesystem_service)
register_exists(mcp, filesystem_service)
register_metadata(mcp, filesystem_service)
register_search_files(mcp, filesystem_service)
register_watch_directory(mcp, filesystem_service)


@mcp.tool
def health() -> dict:
    """Health endpoint."""
    return {
        "status": "healthy",
        "server": "file-server",
        "version": VERSION,
    }


if __name__ == "__main__":
    mcp.run()
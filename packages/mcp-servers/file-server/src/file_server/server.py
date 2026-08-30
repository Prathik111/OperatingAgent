"""
Filesystem MCP Server using FastMCP.

This module creates the FastMCP server and registers all filesystem tools.
Business logic remains inside FileSystemService.
"""

from __future__ import annotations

from pathlib import Path
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

def build_file_server(root: str | Path | None = None) -> FastMCP:
    """Build a file server confined to ``root`` (or the configured default)."""
    server = FastMCP(
        name="file-server",
        version=VERSION,
        mask_error_details=True,
    )
    service = FileSystemService(root=Path(root).expanduser().resolve()) if root else FileSystemService()

    register_read_file(server, service)
    register_write_file(server, service)
    register_delete_file(server, service)
    register_copy_file(server, service)
    register_move_file(server, service)
    register_rename_file(server, service)
    register_list_directory(server, service)
    register_create_directory(server, service)
    register_delete_directory(server, service)
    register_exists(server, service)
    register_metadata(server, service)
    register_search_files(server, service)
    register_watch_directory(server, service)

    @server.tool
    def health() -> dict:
        """Health endpoint."""
        return {
            "status": "healthy",
            "server": "file-server",
            "version": VERSION,
        }

    # Kept for diagnostics and focused tests without exposing service internals
    # through the public tool surface.
    server._operating_agent_filesystem_service = service
    return server


mcp = build_file_server()
filesystem_service = mcp._operating_agent_filesystem_service


if __name__ == "__main__":
    mcp.run()

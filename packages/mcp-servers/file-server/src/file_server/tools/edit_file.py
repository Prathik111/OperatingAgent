"""
FastMCP tool for surgical, anchor-based file edits.
"""

from __future__ import annotations

from fastmcp import FastMCP

from ..services.filesystem_service import FileSystemService


def register_edit_file(
    mcp: FastMCP,
    service: FileSystemService,
) -> None:
    """Register the edit_file MCP tool."""

    @mcp.tool(
        name="edit_file",
        description=(
            "Make a surgical edit to an existing text file by replacing an exact "
            "anchor string, leaving the rest of the file untouched. Prefer this over "
            "write_file for changing part of a large file. The old_string must match "
            "the file exactly, including whitespace and indentation, and must be "
            "unique unless replace_all is set; a non-matching or ambiguous anchor "
            "fails without changing the file, rather than editing the wrong place."
        ),
    )
    def edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        encoding: str = "utf-8",
    ):
        """
        Edit a file by replacing an exact anchor.

        Args:
            path: File path to edit. The file must already exist (use write_file
                to create one).
            old_string: Exact text to find. Must be non-empty and, unless
                replace_all is true, must occur exactly once.
            new_string: Text to put in its place. May be an empty string, which
                deletes the anchor.
            replace_all: Replace every occurrence instead of requiring a single
                unique match.
            encoding: Text encoding used to read and write the file.

        Returns:
            Result from the filesystem service (path, replacements, byte sizes).
        """

        if not path.strip():
            raise ValueError("path must be provided.")

        if not old_string:
            raise ValueError("old_string must be provided and non-empty.")

        if new_string is None:
            raise ValueError(
                "new_string must be provided (use an empty string to delete the anchor)."
            )

        return service.edit_file(
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            encoding=encoding,
        )

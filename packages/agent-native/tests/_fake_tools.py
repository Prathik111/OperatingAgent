"""In-process fake tools, for tests only.

These used to ship as the agent's built-in tools. Now that the agent gets its
real tools from the MCP gateway, they live here purely as test fixtures: they let
the tool-gate tests and the live loop tests exercise both sides of the permission
gate (a read-only tool the policy just allows, a writing tool it stops and asks
about) without needing the whole MCP stack running.

Both are locked to the session's working folder, so a test asking for
`../../etc/passwd` gets a plain error, not a file.
"""

from __future__ import annotations

import os
from typing import Any

from agent_native.tools.base import (
    ExecutionMode,
    Tool,
    ToolDefinition,
    ToolPermissions,
    ToolResult,
)


def _resolve_in_workspace(context: Any, path: str) -> str:
    """Turn a user-supplied path into an absolute one, or refuse if it escapes."""
    root = os.path.realpath(getattr(context.session, "working_directory", ".") or ".")
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(
            f"path {path!r} is outside the working folder; only files under the "
            "session's folder can be reached"
        )
    return target


class ReadFileTool(Tool):
    """Read a text file from the working folder (read-only, so the policy allows it)."""

    _definition = ToolDefinition(
        name="read_file",
        description="Read a text file inside the working folder and return its contents.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the working folder."}},
            "required": ["path"],
        },
        permissions=ToolPermissions(read_only=True, execution_mode=ExecutionMode.DIRECT),
    )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def preview(self, arguments: dict) -> str:
        return f"Read the file {arguments.get('path', '?')!r}"

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        try:
            target = _resolve_in_workspace(context, arguments["path"])
        except (ValueError, KeyError) as exc:
            return ToolResult(False, error=str(exc))
        if not os.path.isfile(target):
            return ToolResult(False, error=f"No such file: {arguments['path']!r}")
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                return ToolResult(True, output=fh.read())
        except OSError as exc:
            return ToolResult(False, error=f"Could not read file: {exc}")


class WriteFileTool(Tool):
    """Write a text file in the working folder (destructive, so the policy asks)."""

    _definition = ToolDefinition(
        name="write_file",
        description="Write (or overwrite) a text file inside the working folder.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working folder."},
                "content": {"type": "string", "description": "The text to write."},
            },
            "required": ["path", "content"],
        },
        permissions=ToolPermissions(destructive=True, execution_mode=ExecutionMode.DIRECT),
    )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def preview(self, arguments: dict) -> str:
        content = arguments.get("content", "")
        return f"Write {len(content)} characters to {arguments.get('path', '?')!r}"

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        try:
            target = _resolve_in_workspace(context, arguments["path"])
        except (ValueError, KeyError) as exc:
            return ToolResult(False, error=str(exc))
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            content = arguments.get("content", "")
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(content)
            return ToolResult(True, output=f"Wrote {len(content)} characters to {arguments['path']!r}")
        except OSError as exc:
            return ToolResult(False, error=f"Could not write file: {exc}")


def default_tools() -> list:
    """The pair of fake tools the gate/loop tests register."""
    return [ReadFileTool(), WriteFileTool()]

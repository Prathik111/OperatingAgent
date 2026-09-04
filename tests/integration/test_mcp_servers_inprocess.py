"""Integration: each real MCP server driven over the real protocol.

The unit tier tests the *services* directly (``FileSystemService``,
``SearchService``, ...) and the agent tier fakes the registry entirely. Neither
proves the piece in between: that each server actually registers its tools, that
the FastMCP schemas match the service signatures, and that a tool invocation
survives the round trip the agent will make in production.

Each server exposes a module-level ``mcp`` object, and ``fastmcp.Client`` accepts
it directly, so all of this runs in-process — no gateway, no sockets, no
subprocess. Marked ``integration`` because real filesystem and git operations
happen (inside the temp ``FILE_SERVER_ROOT`` that ``tests/conftest.py`` pins).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from agent_langgraph.mcp_adapter import MCPAdapter
from agent_langgraph.runtime.tool_registry import ToolRegistry


def registry_for(server) -> ToolRegistry:
    """The same wrapper the executor uses, over a real in-process server."""
    return ToolRegistry(MCPAdapter(server))


# ---------------------------------------------------------------------------
# file-server — real filesystem, confined to the pinned temp root
# ---------------------------------------------------------------------------


async def test_file_server_round_trips_a_write_then_read(file_server_root: Path) -> None:
    from file_server.server import mcp

    registry = registry_for(mcp)
    name = "integration-round-trip.txt"

    written = await registry.call_by_name(
        "write_file", {"path": name, "content": "written over MCP"}
    )
    assert written.success is True, written.error

    # The write really hit the disk the service was pinned to.
    assert (file_server_root / name).read_text(encoding="utf-8") == "written over MCP"

    read = await registry.call_by_name("read_file", {"path": name})
    assert read.success is True, read.error
    assert "written over MCP" in str(read.output)


async def test_file_server_exposes_its_tool_surface() -> None:
    """A schema mismatch between the service and the tool wrapper shows up here."""
    from file_server.server import mcp

    tools = {t.name: t for t in await registry_for(mcp).list_tools()}
    assert {"read_file", "write_file", "list_directory", "exists"} <= set(tools)
    assert "path" in tools["read_file"].schema.input_schema.get("properties", {})


@pytest.mark.regression
async def test_file_server_refuses_to_escape_its_root() -> None:
    """Path confinement is a security control, and it must hold *through* the MCP
    layer — not just in a direct service call."""
    from file_server.server import mcp

    result = await registry_for(mcp).call_by_name(
        "read_file", {"path": "../../../../etc/passwd"}
    )
    assert result.success is False
    assert result.error


# ---------------------------------------------------------------------------
# search-server — in-memory, so entirely deterministic
# ---------------------------------------------------------------------------


async def test_search_server_indexes_then_finds_a_document() -> None:
    from search_server.server import mcp

    registry = registry_for(mcp)

    indexed = await registry.call_by_name(
        "index_documents",
        {
            "index_name": "integration",
            "documents": [
                {"id": "1", "text": "the quick brown fox"},
                {"id": "2", "text": "a slow green turtle"},
            ],
        },
    )
    assert indexed.success is True, indexed.error

    found = await registry.call_by_name(
        "search_documents", {"index_name": "integration", "query": "brown"}
    )
    assert found.success is True, found.error
    assert "fox" in str(found.output)


# ---------------------------------------------------------------------------
# terminal-server — the allowlist is the safety boundary
# ---------------------------------------------------------------------------


async def test_terminal_server_lists_processes() -> None:
    from terminal_server.server import mcp

    result = await registry_for(mcp).call_by_name("list_processes", {})
    assert result.success is True, result.error


@pytest.mark.regression
async def test_terminal_server_rejects_a_command_outside_the_allowlist() -> None:
    """``TERMINAL_SERVER_ALLOWED_COMMANDS`` is pinned by conftest to read-only
    commands. Enforcement must survive the MCP round trip."""
    from terminal_server.server import mcp

    tools = {t.name for t in await registry_for(mcp).list_tools()}
    if "run_command" not in tools:
        pytest.skip("run_command is disabled via TERMINAL_SERVER_ENABLE_RUN_COMMAND")

    result = await registry_for(mcp).call_by_name(
        "run_command", {"command": "rm -rf /"}
    )
    assert result.success is False
    assert result.error


# ---------------------------------------------------------------------------
# git-server — needs the real git binary
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not available")
async def test_git_server_reports_status_for_this_repository() -> None:
    from git_server.server import mcp

    # This repo is a git repository (the suite runs from its root).
    result = await registry_for(mcp).call_by_name("git_status", {"repository": "."})
    assert result.success is True, result.error
    assert result.output is not None

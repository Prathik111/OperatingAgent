"""Built-in tools: there aren't any anymore, on purpose.

The agent used to ship a `read_file` and a `write_file` of its own. It doesn't
now - it gets the real thing (read, write, list, search, git, terminal) from the
MCP gateway instead, wired up in `tools/mcp_bridge.py`. The file server behind the
gateway is a strict superset of the old pair, so nothing was lost by dropping them.

`default_tools()` stays as an empty list so the wiring that used to register the
built-ins still has something to call; it simply registers nothing. The old tools
now live only as test fixtures in `tests/_fake_tools.py`.
"""

from __future__ import annotations


def default_tools() -> list:
    """No built-in tools ship now; the agent's tools come from the MCP gateway."""
    return []

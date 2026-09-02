"""
Gateway MCP Server using FastMCP server composition.

This server does not define domain tools of its own. Instead it *mounts*
every operating-agent sub-server (filesystem, git, terminal, search) behind a
single FastMCP instance, so any MCP client - the native agent's in-memory
provider, a LangGraph agent over stdio/HTTP, or ``fastmcp run`` - reaches the
whole fleet through one connection.

Mounting is a live link (FastMCP 3.x ``mount``; the older ``import_server`` is
deprecated). Each sub-server is given a namespace, so its tools are exposed as
``<namespace>_<tool>`` (e.g. ``filesystem_read_file``, ``git_git_status``).
The namespaces intentionally match the categories used by the native agent's
``tools.enabled`` config, so category filtering lines up end to end.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Final

from fastmcp import FastMCP
from file_server.server import build_file_server
from file_server.server import mcp as file_mcp
from git_server.server import mcp as git_mcp
from search_server.server import mcp as search_mcp
from terminal_server.server import mcp as terminal_mcp

VERSION: Final[str] = "0.1.0"

# namespace -> mounted sub-server. The namespace becomes the tool-name prefix
# and the native agent's tool category, so keep these keys aligned with the
# native config's ``tools.enabled`` list.
MOUNTS: Final[dict[str, FastMCP]] = {
    "filesystem": file_mcp,
    "git": git_mcp,
    "terminal": terminal_mcp,
    "search": search_mcp,
}

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Server lifecycle: log startup/shutdown for the gateway itself."""
    LOGGER.info(
        "gateway-server starting version=%s namespaces=%s",
        VERSION,
        ", ".join(MOUNTS),
    )
    try:
        yield
    finally:
        LOGGER.info("gateway-server shutting down")


def build_gateway(root: str | None = None) -> FastMCP:
    """Create the gateway and mount every sub-server under its namespace.

    Returns:
        A FastMCP instance exposing all mounted sub-server tools plus a
        gateway-level health probe.
    """
    gateway = FastMCP(
        name="operating-agent-gateway",
        version=VERSION,
        instructions=(
            "Operating-agent gateway composing the filesystem, git, terminal "
            "and search MCP servers behind a single endpoint. Every tool is "
            "namespaced by its source server (filesystem_*, git_*, terminal_*, "
            "search_*). Use gateway_health for a single liveness probe across "
            "all mounted servers."
        ),
        lifespan=lifespan,
        mask_error_details=True,
        list_page_size=50,
        strict_input_validation=True,
    )

    mounts = dict(MOUNTS)
    if root is not None:
        mounts["filesystem"] = build_file_server(root=root)
    gateway._operating_agent_mounts = mounts

    for namespace, sub_server in mounts.items():
        gateway.mount(sub_server, namespace=namespace)

    @gateway.tool
    def gateway_health() -> dict:
        """Report gateway liveness and the mounted server namespaces."""
        return {
            "status": "healthy",
            "server": "operating-agent-gateway",
            "version": VERSION,
            "namespaces": list(mounts),
        }

    return gateway


mcp = build_gateway()


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)

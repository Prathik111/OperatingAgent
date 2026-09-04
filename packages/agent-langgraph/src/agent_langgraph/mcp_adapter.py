# use fastmcp to create a mcp adapter for the agent-langgraph package
"""MCPAdapter — the LangGraph track's IMCPClient implementation.

Wraps a FastMCP ``Client`` pointed at the gateway server and maps the MCP
wire types (``mcp.types.Tool``, ``CallToolResult``) onto the project's own
``ToolInfo`` / ``ToolCallResult`` so the rest of the agent never depends on
FastMCP directly. ToolRegistry sits on top of this.
"""

from __future__ import annotations

import logging
from typing import Any

from common.interfaces import IMCPClient
from common.tools import ToolCallRequest, ToolCallResult, ToolInfo, ToolSchema
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

log = logging.getLogger(__name__)

class MCPAdapter(IMCPClient):
    """IMCPClient over a FastMCP gateway.

    ``transport`` is anything ``fastmcp.Client`` accepts — a gateway URL
    string, an in-process ``FastMCP`` instance (handy for tests), an
    ``MCPConfig``/dict, or a path. A fresh session is opened per operation via
    ``async with``, which is the safe, reconnect-tolerant FastMCP pattern.
    """

    def __init__(
        self,
        transport: Any,
        *,
        stdio_command: str | None = None,
        stdio_args: list[str] | None = None,
    ) -> None:
        self._client = Client(transport)
        self._stdio_command = stdio_command
        self._stdio_args = list(stdio_args or [])

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
    ) -> MCPAdapter:
        """Connect to a FastMCP server launched as a subprocess over stdio."""
        return cls(
            StdioTransport(command=command, args=args, cwd=cwd),
            stdio_command=command,
            stdio_args=args,
        )

    def for_workspace(self, workspace: str) -> MCPAdapter:
        """Create an equivalent stdio client rooted at one task workspace."""
        if self._stdio_command is None:
            return self
        return self.from_stdio(
            self._stdio_command,
            list(self._stdio_args),
            cwd=workspace,
        )

    async def aclose(self) -> None:
        """Close the client and terminate a keep-alive stdio subprocess."""
        try:
            await self._client.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must remain best effort
            log.debug("MCP client shutdown failed: %s", exc)
            try:
                await self._client.transport.close()
            except Exception as transport_exc:  # noqa: BLE001
                log.debug("MCP transport shutdown failed: %s", transport_exc)

    async def list_tools(self) -> list[ToolInfo]:
        """List gateway tools, mapped to the project's ToolInfo."""
        async with self._client:
            tools = await self._client.list_tools()
        return [self._to_tool_info(t) for t in tools]

    async def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """Invoke a tool and normalise the outcome to ToolCallResult.

        Uses ``raise_on_error=False`` so a tool-reported error surfaces as
        ``success=False`` with the message, rather than raising; transport /
        connection failures still raise and are caught here as a failed call.
        """
        try:
            async with self._client:
                result = await self._client.call_tool(
                    request.tool_name,
                    request.arguments,
                    raise_on_error=False,
                )
        except Exception as exc:  # noqa: BLE001 - external transport boundary
            log.warning("MCP call to %s failed at transport: %s", request.tool_name, exc)
            return ToolCallResult(success=False, output=None, error=str(exc))

        if result.is_error:
            message = self._error_text(result)
            return ToolCallResult(success=False, output=None, error=message)

        # Prefer the deserialized structured output; fall back to raw content.
        output = result.data
        if output is None:
            output = result.structured_content if result.structured_content is not None else result.content
        return ToolCallResult(success=True, output=output, error=None)

    # -- mapping helpers ---------------------------------------------------

    @staticmethod
    def _to_tool_info(tool: Any) -> ToolInfo:
        risk = "safe"
        annotations = getattr(tool, "annotations", None)
        if annotations is not None:
            # Some servers advertise a risk hint via annotations.
            risk = getattr(annotations, "risk_level", None) or risk

        return ToolInfo(
            name=tool.name,
            description=tool.description or "",
            schema=ToolSchema(
                input_schema=getattr(tool, "inputSchema", None) or {},
                output_schema=getattr(tool, "outputSchema", None) or {},
            ),
            risk_level=risk,
        )

    @staticmethod
    def _error_text(result: Any) -> str:
        """Extract a human-readable error from a CallToolResult's content."""
        content = getattr(result, "content", None)
        if content:
            first = content[0]
            text = getattr(first, "text", None)
            if text:
                return text
        return "tool reported an error"

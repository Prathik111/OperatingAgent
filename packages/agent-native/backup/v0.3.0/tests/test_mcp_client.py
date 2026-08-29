"""NativeMCPClient tests against a real fake server subprocess.

Covers both framing styles (length-prefixed per FastMCP 2.x and
newline-delimited), the initialize handshake, tools/list mapping, tools/call
success + error contracts, transport failure, and close().
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from agent_native.mcp import StdioMCPClient
from agent_native.types import ToolCallRequest

FAKE_SERVER = Path(__file__).resolve().parent / "fake_mcp_server.py"


async def _client(framing: str) -> StdioMCPClient:
    async def spawn():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(FAKE_SERVER), "--framing", framing,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        return proc.stdout, proc.stdin, proc

    return StdioMCPClient(name="fake", spawner=spawn, request_timeout_s=10, connect_timeout_s=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["length", "newline"])
async def test_framing_styles(framing: str):
    client = await _client(framing)
    await client.connect()
    assert client._connected
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["length", "newline"])
async def test_list_tools(framing: str):
    client = await _client(framing)
    try:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {"echo", "boom"} <= names
        echo = next(t for t in tools if t.name == "echo")
        assert echo.schema.input_schema.get("required") == ["text"]
        assert "Echo" in echo.description
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["length", "newline"])
async def test_call_tool_success(framing: str):
    client = await _client(framing)
    try:
        result = await client.call_tool(ToolCallRequest(tool_name="echo", arguments={"text": "hi"}))
        assert result.success is True
        assert "echoed:hi" in str(result.output)
        assert result.error is None
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["length", "newline"])
async def test_call_tool_error_returns_result_not_exception(framing: str):
    client = await _client(framing)
    try:
        result = await client.call_tool(ToolCallRequest(tool_name="boom", arguments={}))
        assert result.success is False
        assert "boom failed" in (result.error or "")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unknown_tool_error():
    client = await _client("length")
    try:
        result = await client.call_tool(ToolCallRequest(tool_name="nope", arguments={}))
        assert result.success is False
        assert "unknown tool" in (result.error or "")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_server_close_returns_failed_result_not_exception():
    """Contract: call_tool never raises - transport failures become
    ToolCallResult(success=False, error=...)."""
    async def spawn():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(FAKE_SERVER), "--framing", "length",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        return proc.stdout, proc.stdin, proc

    client = StdioMCPClient(name="fake", spawner=spawn, request_timeout_s=5)
    await client.connect()
    assert client._proc is not None
    client._proc.kill()
    await client._proc.wait()
    result = await client.call_tool(ToolCallRequest(tool_name="echo", arguments={}))
    assert result.success is False
    assert "mcp transport failure" in (result.error or "")
    await client.close()


@pytest.mark.asyncio
async def test_spawn_failure_raises_transport_error():
    async def bad_spawn():
        raise FileNotFoundError("no such binary")

    client = StdioMCPClient(name="fake", spawner=bad_spawn)
    from agent_native.mcp import MCPTransportError

    with pytest.raises(MCPTransportError):
        await client.connect()
    await client.close()
"""NativeMCPClient - a hand-rolled MCP protocol client over stdio.

Speaks JSON-RPC 2.0 with the MCP stdio framing: initialize handshake,
`notifications/initialized`, `tools/list`, `tools/call`. Two framing styles
are supported transparently, matching FastMCP 2.x (Content-Length headers,
revision 2025-03-26) and the older newline-delimited JSON revision.

Contract:
  - `list_tools()` / `call_tool()` return local types (ToolInfo,
    ToolCallResult) - they never raise for server-side failures; a failed
    call returns ToolCallResult(success=False, error=...).
  - Each instance owns exactly one child process; close() terminates it.

Spawn: the caller injects an async `spawner` callable returning
(reader, writer, process). Default spawns `python -m <name>` directly; the
sandbox wiring (agent.py) injects a spawner that runs the server inside the
task's Docker container (decision #6 - same container, same egress policy).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Awaitable, Callable, Protocol

from ..types import ToolCallRequest, ToolCallResult, ToolInfo, ToolSchema

PROTOCOL_VERSION = "2025-03-26"

MCP_PROTOCOL_ERROR = -32603

Spawner = Callable[[], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter, subprocess.Popen]]]


class MCPClient(Protocol):
    """Local protocol - deliberately agent-native's own, not common's."""

    async def list_tools(self) -> list[ToolInfo]: ...

    async def call_tool(self, request: ToolCallRequest) -> ToolCallResult: ...

    async def close(self) -> None: ...


class MCPTransportError(RuntimeError):
    pass


class _StdioReader:
    """Incremental MCP frame reader supporting both framing styles."""

    def __init__(self, stream: asyncio.StreamReader) -> None:
        self._stream = stream
        self._buffer = bytearray()

    async def read_frame(self) -> dict:
        while True:
            frame = self._try_parse()
            if frame is not None:
                return frame
            chunk = await self._stream.read(65536)
            if not chunk:
                if self._try_parse() is not None:
                    return self._try_parse()
                raise MCPTransportError("MCP server closed stdout unexpectedly")
            self._buffer.extend(chunk)

    async def drain(self) -> None:
        """Read remaining buffered output to EOF so the pipe transport is
        released before the loop closes (avoids Windows proactor noise)."""
        while await self._stream.read(65536):
            pass

    def _try_parse(self) -> dict | None:
        buf = bytes(self._buffer)
        header_end = buf.find(b"\r\n\r\n")
        if header_end != -1:
            header = buf[:header_end].decode("utf-8", "replace")
            length = None
            for line in header.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip())
            if length is not None and len(buf) >= header_end + 4 + length:
                payload = buf[header_end + 4: header_end + 4 + length]
                del self._buffer[: header_end + 4 + length]
                return json.loads(payload)
        newline = buf.find(b"\n")
        if newline != -1:
            line = buf[:newline].strip()
            del self._buffer[: newline + 1]
            if not line:
                return None
            return json.loads(line)
        return None


async def _default_spawner(name: str) -> Spawner:
    async def _spawn() -> tuple[asyncio.StreamReader, asyncio.StreamWriter, subprocess.Popen]:
        proc = await asyncio.create_subprocess_exec(
            *["python", "-m", name],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        return proc.stdout, proc.stdin, proc

    return _spawn


class StdioMCPClient:
    """MCP client over a stdio child process (JSON-RPC 2.0)."""

    def __init__(
        self,
        name: str = "",
        spawner: Spawner | None = None,
        *,
        request_timeout_s: float = 30.0,
        connect_timeout_s: float = 15.0,
    ) -> None:
        self.name = name
        self._spawner = spawner
        self._request_timeout_s = request_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._proc: subprocess.Popen | None = None
        self._reader: _StdioReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._next_id = 1
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return
        if self._spawner is None:
            self._spawner = await _default_spawner(self.name)
        try:
            raw_reader, self._writer, self._proc = await asyncio.wait_for(
                self._spawner(), timeout=self._connect_timeout_s
            )
            self._reader = _StdioReader(raw_reader)
        except Exception as e:
            raise MCPTransportError(f"failed to spawn MCP server {self.name!r}: {e}") from e
        try:
            await self._handshake()
            self._connected = True
        except Exception:
            await self.close()
            raise

    async def _handshake(self) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agent-native", "version": "0.3.0"},
            },
        )
        if "result" not in result:
            raise MCPTransportError(f"initialize failed: {json.dumps(result)[:300]}")

    async def list_tools(self) -> list[ToolInfo]:
        await self.connect()
        result = await self._request("tools/list", {})
        r = result.get("result") or {}
        tools: list[ToolInfo] = []
        for t in r.get("tools", []):
            schema = t.get("inputSchema") or t.get("schema") or {}
            tools.append(ToolInfo(
                name=t.get("name", ""),
                description=t.get("description", ""),
                schema=ToolSchema(
                    input_schema=schema,
                    output_schema=t.get("outputSchema", {}),
                ),
                risk_level=t.get("riskLevel", "safe"),
            ))
        return tools

    async def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        await self.connect()
        try:
            result = await self._request("tools/call", {"name": request.tool_name, "arguments": request.arguments})
        except (MCPTransportError, ConnectionError, BrokenPipeError, OSError) as e:
            return ToolCallResult(success=False, output=None, error=f"mcp transport failure: {e}")
        r = result.get("result") or {}
        if "error" in r and r.get("isError"):
            return ToolCallResult(success=False, output=None, error=str(r.get("error")))
        if isinstance(r.get("content"), list) and r["content"]:
            text = " ".join(
                c.get("text", json.dumps(c)) for c in r["content"] if isinstance(c, dict)
            )
        else:
            text = r.get("structuredContent") or r.get("content") or ""
        success = not bool(r.get("isError", False))
        error = r.get("error") if not success else None
        return ToolCallResult(success=success, output=text, error=error)

    async def _notify(self, method: str, params: dict) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, method: str, params: dict) -> dict:
        msg_id = self._next_id
        self._next_id += 1
        await self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        while True:
            frame = await self._read_with_timeout()
            if frame.get("method"):
                if frame.get("id") is not None:
                    await self._send({
                        "jsonrpc": "2.0",
                        "id": frame["id"],
                        "error": {"code": MCP_PROTOCOL_ERROR, "message": "method not found"},
                    })
                continue
            if frame.get("id") == msg_id:
                return frame

    async def _read_with_timeout(self) -> dict:
        assert self._reader is not None
        return await asyncio.wait_for(
            self._reader.read_frame(), timeout=self._request_timeout_s
        )

    async def _send(self, payload: dict) -> None:
        assert self._writer is not None
        self._writer.write(json.dumps(payload).encode("utf-8") + b"\n")
        await self._writer.drain()

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        proc = self._proc
        if proc is not None and _is_running(proc):
            try:
                proc.terminate()
                if isinstance(proc, asyncio.subprocess.Process):
                    await asyncio.wait_for(proc.wait(), timeout=5)
                else:
                    await asyncio.get_event_loop().run_in_executor(None, proc.wait, 5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._reader is not None:
            try:
                await asyncio.wait_for(self._reader.drain(), timeout=2.0)
            except Exception:
                pass
        self._connected = False


def _is_running(proc) -> bool:
    poll = getattr(proc, "poll", None)
    if poll is not None:
        return poll() is None
    return proc.returncode is None

"""Fake MCP server for StdioMCPClient tests.

Auto-detects framing per incoming message - length-prefixed (Content-Length
headers, MCP 2025-03-26, what the real FastMCP 2.x servers use) or
newline-delimited JSON (legacy revision). Responds in the same style the
request arrived in. `--framing length|newline` forces a response style.

Speaks just enough of the MCP protocol: initialize handshake, tools/list,
tools/call (echo + failure), and answers unknown methods with a JSON-RPC
error (exercising the client's protocol-error reply path).

Usage: python fake_mcp_server.py [--framing length|newline|auto]
"""

from __future__ import annotations

import json
import sys

FORCE_FRAMING = "auto"
if "--framing" in sys.argv:
    FORCE_FRAMING = sys.argv[sys.argv.index("--framing") + 1]


def _read_message() -> dict | None:
    first = sys.stdin.buffer.readline()
    if not first:
        return None
    if first.lower().startswith(b"content-length:"):
        length = int(first.split(b":", 1)[1].strip())
        while True:
            line = sys.stdin.buffer.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        body = sys.stdin.buffer.read(length)
        return json.loads(body)
    line = first.strip()
    if not line:
        return None
    return json.loads(line)


def _write_message(obj: dict, framing: str) -> None:
    body = json.dumps(obj).encode()
    if framing == "length":
        sys.stdout.buffer.write(
            b"Content-Length: " + str(len(body)).encode()
            + b"\r\nContent-Type: application/json\r\n\r\n" + body
        )
    else:
        sys.stdout.buffer.write(body + b"\n")
    sys.stdout.buffer.flush()


def _detect_framing() -> str:
    if FORCE_FRAMING != "auto":
        return FORCE_FRAMING
    first = sys.stdin.buffer.peek(64)  # type: ignore[attr-defined]
    return "length" if first.lower().startswith(b"content-length:") else "newline"


def main() -> None:
    framing = _detect_framing()
    while True:
        msg = _read_message()
        if msg is None:
            return
        method = msg.get("method")
        msg_id = msg.get("id")
        if msg_id is None:
            continue  # notification
        if method == "initialize":
            _write_message({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp-server", "version": "0.1.0"},
                },
            }, framing)
        elif method == "tools/list":
            _write_message({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": [
                    {
                        "name": "echo",
                        "description": "Echo arguments back",
                        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                                        "required": ["text"]},
                    },
                    {
                        "name": "boom",
                        "description": "Always fails",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]},
            }, framing)
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name")
            if name == "echo":
                result = {"content": [{"type": "text", "text": f"echoed:{params.get('arguments', {}).get('text', '')}"}]}
            elif name == "boom":
                result = {"isError": True, "error": "boom failed", "content": []}
            else:
                result = {"isError": True, "error": f"unknown tool {name}", "content": []}
            _write_message({"jsonrpc": "2.0", "id": msg_id, "result": result}, framing)
        else:
            _write_message({"jsonrpc": "2.0", "id": msg_id,
                            "error": {"code": -32601, "message": "method not found"}}, framing)


if __name__ == "__main__":
    main()
"""The MCP bridge: name -> permissions, reply -> ToolResult, and the tool gate.

These run offline. They never import fastmcp or the gateway: an MCPTool is built
directly around a tiny fake client, which is all `execute` needs. The one thing
that truly needs fastmcp - opening a real connection - is only checked for its
friendly "not installed" message, and that check skips itself where fastmcp is
present.
"""

from __future__ import annotations

import asyncio

from agent_native.config import AgentConfig
from agent_native.conversation import Session, ToolCall
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.loop import Cancellation, Limits, RunContext
from agent_native.permissions import (
    PermissionManager,
    PermissionStore,
    PolicyChain,
    RulePolicy,
    SessionPolicy,
)
from agent_native.tools.base import ToolRegistry
from agent_native.tools.manager import ToolManager
from agent_native.tools.mcp_bridge import (
    MCPTool,
    MCPToolProvider,
    _infer_permissions,
    _to_tool_result,
)


# ---------------------------------------------------------------------------
# Fakes: enough of FastMCP's shapes to exercise the bridge, nothing more.
# ---------------------------------------------------------------------------
class FakeSpec:
    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class FakeText:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeCallResult:
    def __init__(self, content=None, data=None, structured_content=None, is_error=False):
        self.content = content or []
        self.data = data
        self.structured_content = structured_content
        self.is_error = is_error


class FakeClient:
    """Records each call and returns a scripted result (or raises)."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._raises is not None:
            raise self._raises
        return self._result


def _read_tool(client=None):
    return MCPTool(
        client or FakeClient(),
        FakeSpec(
            "filesystem_read_file",
            "Read a file.",
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
    )


# ---------------------------------------------------------------------------
# spec -> ToolDefinition
# ---------------------------------------------------------------------------
def test_spec_becomes_definition():
    d = _read_tool().definition
    assert d.full_name == "filesystem_read_file"   # namespace empty, name is exact
    assert d.name == "filesystem_read_file"
    assert d.description == "Read a file."
    assert d.input_schema["required"] == ["path"]
    assert d.permissions.read_only is True


def test_spec_from_dict_shape():
    tool = MCPTool(
        FakeClient(),
        {"name": "gateway_health", "description": "probe", "inputSchema": {"type": "object"}},
    )
    assert tool.definition.full_name == "gateway_health"
    assert tool.definition.description == "probe"
    assert tool.definition.permissions.read_only is True


# ---------------------------------------------------------------------------
# name -> permission flags
# ---------------------------------------------------------------------------
def test_permission_inference_table():
    p = _infer_permissions
    # filesystem reads vs writes
    assert p("filesystem_read_file").read_only
    assert p("filesystem_list_directory").read_only
    assert p("filesystem_metadata").read_only
    assert p("filesystem_write_file").destructive
    assert p("filesystem_delete_directory").destructive
    assert p("filesystem_move_file").destructive
    # git: every tool the gateway exposes today is a read
    for name in ("git_diff", "git_git_log", "git_git_status", "git_list_branches"):
        assert p(name).read_only, name
    # git mutations / network - guarding the future, not exposed yet
    assert p("git_commit").destructive
    assert p("git_push").destructive and p("git_push").needs_network
    # terminal. The mode is `sandbox`, not `host_process`: a shell command is the
    # one call that should run inside a container when there is one, and the flag
    # is how the tool manager knows to route it there (see tools/sandbox.py). It
    # still asks first either way - `destructive` is what does that.
    shell = p("terminal_run_command")
    assert shell.destructive and shell.execution_mode.value == "sandbox"
    assert p("terminal_list_processes").read_only
    # search
    assert p("search_search_documents").read_only
    assert p("search_list_indices").read_only
    index = p("search_index_documents")
    assert not index.read_only and not index.destructive   # in-memory write -> policy asks
    # gateway health, and an unknown server
    assert p("gateway_health").read_only
    unknown = p("weather_forecast")
    assert not unknown.read_only and not unknown.destructive


# ---------------------------------------------------------------------------
# reply -> ToolResult, across the shapes different fastmcp versions return
# ---------------------------------------------------------------------------
def test_result_prefers_text_content():
    r = _to_tool_result(FakeCallResult(content=[FakeText("file body")]))
    assert r.success and r.output == "file body"


def test_result_falls_back_to_structured():
    r = _to_tool_result(FakeCallResult(structured_content={"count": 2}))
    assert r.success and '"count"' in r.output


def test_result_falls_back_to_data():
    r = _to_tool_result(FakeCallResult(data=42))
    assert r.success and r.output == "42"


def test_result_error_flag_becomes_failure():
    r = _to_tool_result(FakeCallResult(content=[FakeText("boom")], is_error=True))
    assert not r.success and "boom" in r.error


def test_result_bare_list_is_supported():
    r = _to_tool_result([FakeText("legacy shape")])
    assert r.success and r.output == "legacy shape"


# ---------------------------------------------------------------------------
# execute: forwards the call, maps the reply, never throws
# ---------------------------------------------------------------------------
async def test_execute_forwards_and_maps():
    client = FakeClient(result=FakeCallResult(content=[FakeText("port=8080")]))
    result = await _read_tool(client).execute({"path": "config.txt"}, context=None)
    assert result.success and result.output == "port=8080"
    assert client.calls == [("filesystem_read_file", {"path": "config.txt"})]


async def test_execute_turns_an_exception_into_a_result():
    client = FakeClient(raises=RuntimeError("gateway down"))
    result = await _read_tool(client).execute({"path": "x"}, context=None)
    assert not result.success and "gateway down" in result.error


# ---------------------------------------------------------------------------
# the real permission gate reads the inferred flags
# ---------------------------------------------------------------------------
def _manager_with(tool):
    db = MemoryDatabase()
    registry = ToolRegistry()
    registry.register(tool)
    permissions = PermissionManager(PermissionStore(db), EventBus(db))
    policy = PolicyChain([RulePolicy(), SessionPolicy()])
    return ToolManager(registry, policy, permissions), permissions


def _ctx():
    return RunContext(
        session=Session(agent="build", working_directory="."),
        run_id="r",
        config=AgentConfig(),
        limits=Limits(),
        cancellation=Cancellation(),
    )


def _write_tool(client):
    return MCPTool(
        client,
        FakeSpec(
            "filesystem_write_file",
            "Write a file.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        ),
    )


def _shell_tool(client):
    return MCPTool(
        client,
        FakeSpec(
            "terminal_run_command",
            "Run a shell command.",
            {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        ),
    )


async def test_read_tool_runs_without_asking():
    client = FakeClient(result=FakeCallResult(content=[FakeText("ok")]))
    manager, permissions = _manager_with(_read_tool(client))
    result = await manager.execute(
        ToolCall(id="c", name="filesystem_read_file", arguments={"path": "a"}), _ctx()
    )
    assert result.success and not permissions.pending()


async def test_write_tool_is_asked_before_it_runs():
    client = FakeClient(result=FakeCallResult(content=[FakeText("wrote")]))
    manager, permissions = _manager_with(_write_tool(client))
    call = ToolCall(id="c", name="filesystem_write_file", arguments={"path": "o.txt", "content": "hi"})

    async def approve():
        while True:
            if permissions.pending():
                await permissions.resolve("c", True)
                return
            await asyncio.sleep(0.005)

    task = asyncio.create_task(approve())
    result = await manager.execute(call, _ctx())
    await task

    assert result.success
    assert client.calls == [("filesystem_write_file", {"path": "o.txt", "content": "hi"})]


async def test_shell_tool_denied_never_runs():
    client = FakeClient(result=FakeCallResult(content=[FakeText("should not run")]))
    manager, permissions = _manager_with(_shell_tool(client))
    call = ToolCall(id="c", name="terminal_run_command", arguments={"command": "rm -rf /"})

    async def deny():
        while True:
            if permissions.pending():
                await permissions.resolve("c", False)
                return
            await asyncio.sleep(0.005)

    task = asyncio.create_task(deny())
    result = await manager.execute(call, _ctx())
    await task

    assert not result.success and "Not allowed" in result.error
    assert client.calls == []  # denied: the gateway was never touched


# ---------------------------------------------------------------------------
# provider plumbing that doesn't need a live connection
# ---------------------------------------------------------------------------
def test_build_gateway_uses_the_factory():
    sentinel = object()
    provider = MCPToolProvider(gateway_factory=lambda: sentinel)
    assert provider._build_gateway() is sentinel


def test_real_gateway_accepts_and_confines_the_workspace_root(tmp_path):
    from gateway_server.server import build_gateway

    gateway = build_gateway(root=str(tmp_path))
    mounted = gateway._operating_agent_mounts["filesystem"]
    service = mounted._operating_agent_filesystem_service
    assert service.root == tmp_path.resolve()


async def test_close_without_connect_is_safe():
    await MCPToolProvider().close()  # no connection open; must not raise


async def test_import_client_guides_when_fastmcp_missing():
    try:
        import fastmcp  # noqa: F401
        return  # fastmcp is installed here, so the friendly-error path isn't reachable
    except ImportError:
        pass
    from agent_native.tools.mcp_bridge import _import_client

    try:
        _import_client()
    except RuntimeError as exc:
        assert "fastmcp" in str(exc) and "uv sync --all-packages" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError about fastmcp")

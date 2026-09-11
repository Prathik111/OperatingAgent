"""Unit tests for ``ToolRegistry`` (``agent_langgraph.runtime.tool_registry``).

The registry is the thin abstraction the executor depends on instead of
individual MCP servers, so these tests pin delegation: what reaches the adapter
and what comes back. The adapter itself is covered against a real in-process
FastMCP server in ``tests/integration/test_mcp_adapter_fastmcp.py``.
"""

from __future__ import annotations

from agent_langgraph.runtime.tool_registry import ToolRegistry
from common.config import SandboxConfig, ToolPermissionConfig
from common.tools import ToolCallRequest, ToolCallResult, ToolInfo, ToolSchema

from tests.support.langgraph import make_tool_info


class FakeAdapter:
    """Duck-typed stand-in for ``MCPAdapter`` recording what it received."""

    def __init__(self, tools=None, result=None) -> None:
        self._tools = tools or []
        self._result = result or ToolCallResult(success=True, output="ok", error=None)
        self.list_calls = 0
        self.received: list[ToolCallRequest] = []

    async def list_tools(self):
        self.list_calls += 1
        return self._tools

    async def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        self.received.append(request)
        return self._result


async def test_registry_list_tools_delegates() -> None:
    adapter = FakeAdapter(tools=[make_tool_info("echo")])
    registry = ToolRegistry(adapter)
    tools = await registry.list_tools()
    assert [t.name for t in tools] == ["echo"]
    assert adapter.list_calls == 1


async def test_registry_call_delegates_request() -> None:
    adapter = FakeAdapter(result=ToolCallResult(success=True, output="done", error=None))
    registry = ToolRegistry(adapter)
    request = ToolCallRequest(tool_name="echo", arguments={"text": "hi"})
    result = await registry.call(request)
    assert result.output == "done"
    assert adapter.received == [request]


async def test_registry_call_by_name_builds_request() -> None:
    adapter = FakeAdapter()
    registry = ToolRegistry(adapter)
    await registry.call_by_name("echo", {"text": "hi"})
    sent = adapter.received[0]
    assert sent.tool_name == "echo"
    assert sent.arguments == {"text": "hi"}


async def test_registry_filters_and_blocks_disabled_tool_categories() -> None:
    adapter = FakeAdapter(
        tools=[
            make_tool_info("terminal_run_command"),
            make_tool_info("run_command"),
            make_tool_info("git_git_status"),
        ]
    )
    registry = ToolRegistry(
        adapter,
        permissions=ToolPermissionConfig(terminal=False),
        sandbox=SandboxConfig(enabled=False),
    )

    assert [tool.name for tool in await registry.list_tools()] == ["git_git_status"]
    result = await registry.call_by_name("terminal_run_command", {"command": "pwd"})

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert adapter.received == []


async def test_registry_confines_filesystem_paths_to_workspace(tmp_path) -> None:
    adapter = FakeAdapter()
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
    )

    allowed = await registry.call_by_name("filesystem_read_file", {"path": "ok.txt"})
    blocked = await registry.call_by_name("filesystem_read_file", {"path": "../escape.txt"})
    blocked_root = await registry.call_by_name(
        "filesystem_search_files", {"root": "../outside", "query": "x"}
    )

    assert allowed.success is True
    assert blocked.success is False
    assert blocked_root.success is False
    assert "escapes" in (blocked.error or "")
    assert [request.arguments for request in adapter.received] == [{"path": "ok.txt"}]


async def test_registry_confines_uncategorized_tool_paths(tmp_path) -> None:
    adapter = FakeAdapter()
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
    )

    allowed = await registry.call_by_name("custom_read", {"path": "ok.txt"})
    blocked = await registry.call_by_name("custom_read", {"path": "../escape.txt"})

    assert allowed.success is True
    assert blocked.success is False
    assert "escapes" in (blocked.error or "")


async def test_registry_allows_uncategorized_paths_for_explicitly_isolated_adapter(tmp_path) -> None:
    adapter = FakeAdapter()
    adapter.workspace_isolated = True
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
    )

    result = await registry.call_by_name("custom_read", {"path": "../outside.txt"})

    assert result.success is True


async def test_registry_expands_user_paths_before_workspace_validation(tmp_path) -> None:
    adapter = FakeAdapter()
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
    )

    result = await registry.call_by_name("custom_read", {"path": "~/.ssh/id_rsa"})

    assert result.success is False
    assert "escapes" in (result.error or "")
    assert adapter.received == []


async def test_registry_uses_schema_path_fields_for_uncategorized_tools(tmp_path) -> None:
    tool = ToolInfo(
        name="custom_command",
        description="custom command",
        schema=ToolSchema(
            input_schema={
                "type": "object",
                "properties": {"cwd": {"type": "string"}},
            },
            output_schema={},
        ),
    )
    adapter = FakeAdapter(tools=[tool])
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
    )

    result = await registry.call_by_name(
        "custom_command", {"cwd": "../outside"}
    )

    assert result.success is False
    assert "escapes" in (result.error or "")
    assert adapter.received == []


async def test_registry_uses_explicit_tool_path_field_policy(tmp_path) -> None:
    adapter = FakeAdapter()
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
        tool_path_fields={"custom_open": {"location"}},
    )

    result = await registry.call_by_name(
        "custom_open", {"location": "../outside.txt"}
    )

    assert result.success is False
    assert "escapes" in (result.error or "")
    assert adapter.received == []


async def test_registry_uses_array_item_path_format(tmp_path) -> None:
    tool = ToolInfo(
        name="custom_batch",
        description="custom batch operation",
        schema=ToolSchema(
            input_schema={
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "items": {"type": "string", "format": "path"},
                    }
                },
            },
            output_schema={},
        ),
    )
    adapter = FakeAdapter(tools=[tool])
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
    )

    result = await registry.call_by_name(
        "custom_batch", {"targets": ["inside.txt", "../outside.txt"]}
    )

    assert result.success is False
    assert "escapes" in (result.error or "")
    assert adapter.received == []


async def test_registry_discovers_paths_nested_under_array_items(tmp_path) -> None:
    tool = ToolInfo(
        name="custom_copy_many",
        description="copy many files",
        schema=ToolSchema(
            input_schema={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string", "format": "path"},
                            },
                        },
                    }
                },
            },
            output_schema={},
        ),
    )
    adapter = FakeAdapter(tools=[tool])
    registry = ToolRegistry(
        adapter,
        sandbox=SandboxConfig(enabled=True, workspace=tmp_path),
    )

    result = await registry.call_by_name(
        "custom_copy_many", {"files": [{"target": "../outside.txt"}]}
    )

    assert result.success is False
    assert "escapes" in (result.error or "")
    assert adapter.received == []

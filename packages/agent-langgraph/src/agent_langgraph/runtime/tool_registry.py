from pathlib import Path
from typing import Any, ClassVar, cast

from common.config import SandboxConfig, ToolPermissionConfig
from common.interfaces import IMCPClient
from common.tools import ToolCallRequest, ToolCallResult, ToolInfo
from sandbox import DEFAULT_IMAGE, ContainerPool


class ToolRegistry:
    """
    Maps agent tool requests to MCP tools.

    The ExecutorNode should depend on this abstraction,
    not directly on individual MCP servers.
    """

    _CATEGORY_FIELDS: ClassVar[dict[str, str]] = {
        "filesystem": "file_system",
        "terminal": "terminal",
        "git": "git",
        "search": "search",
        "knowledge": "knowledge",
        "memory": "memory",
    }
    _DIRECT_TOOL_CATEGORIES: ClassVar[dict[str, str]] = {
        "read_file": "filesystem",
        "write_file": "filesystem",
        "delete_file": "filesystem",
        "copy_file": "filesystem",
        "move_file": "filesystem",
        "rename_file": "filesystem",
        "list_directory": "filesystem",
        "create_directory": "filesystem",
        "delete_directory": "filesystem",
        "exists": "filesystem",
        "metadata": "filesystem",
        "search_files": "filesystem",
        "watch_directory": "filesystem",
        "run_command": "terminal",
        "list_processes": "terminal",
        "git_status": "git",
        "list_branches": "git",
        "git_log": "git",
        "diff": "git",
        "index_documents": "search",
        "search_documents": "search",
        "list_indices": "search",
    }
    _PATH_KEYS = ("path", "directory", "source", "destination", "root")

    def __init__(
        self,
        mcp_adapter: IMCPClient,
        permissions: ToolPermissionConfig | None = None,
        sandbox: SandboxConfig | None = None,
    ) -> None:
        self._mcp = mcp_adapter
        self._permissions = permissions or ToolPermissionConfig()
        self._sandbox = sandbox or SandboxConfig()
        self._sandbox_pool: ContainerPool | None = None
        self._workspace_clients: dict[str, IMCPClient] = {}

    async def list_tools(self) -> list[ToolInfo]:
        tools = await self._mcp.list_tools()
        return [tool for tool in tools if self._is_allowed(tool.name)]

    async def call(
        self,
        request: ToolCallRequest,
        *,
        workspace: str | None = None,
    ) -> ToolCallResult:
        if not self._is_allowed(request.tool_name):
            category = self._category(request.tool_name) or "unknown"
            return ToolCallResult(
                success=False,
                output=None,
                error=f"tool category '{category}' is disabled by configuration",
            )
        sandbox_error = self._sandbox_error(request, workspace)
        if sandbox_error:
            return ToolCallResult(success=False, output=None, error=sandbox_error)
        sandbox_result = await self._call_in_sandbox(request, workspace)
        if sandbox_result is not None:
            return sandbox_result
        client = self._client_for_workspace(workspace)
        return await client.call_tool(self._request_for_workspace(request, workspace))

    async def call_by_name(
        self,
        tool_name: str,
        arguments: dict,
        *,
        workspace: str | None = None,
    ) -> ToolCallResult:

        request = ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments,
        )

        return await self.call(request, workspace=workspace)

    async def aclose(self) -> None:
        """Close resources owned by the underlying MCP client."""
        if self._sandbox_pool is not None:
            await self._sandbox_pool.stop_all()
        for client in self._workspace_clients.values():
            close_workspace_client = getattr(client, "aclose", None)
            if close_workspace_client is not None:
                await close_workspace_client()
        self._workspace_clients.clear()
        close = getattr(self._mcp, "aclose", None)
        if close is not None:
            await close()

    def _client_for_workspace(self, workspace: str | None) -> IMCPClient:
        if not workspace:
            return self._mcp
        root = str(Path(workspace).expanduser().resolve())
        existing = self._workspace_clients.get(root)
        if existing is not None:
            return existing
        factory = getattr(self._mcp, "for_workspace", None)
        if not callable(factory):
            return self._mcp
        # ``for_workspace`` is an optional extension on IMCPClient.  Keep the
        # base protocol small and validate the dynamic result at this boundary.
        client = cast(IMCPClient, factory(root))
        self._workspace_clients[root] = client
        return client

    @classmethod
    def _category(cls, tool_name: str) -> str | None:
        prefix, separator, _rest = tool_name.partition("_")
        if separator and prefix in cls._CATEGORY_FIELDS:
            return prefix
        return cls._DIRECT_TOOL_CATEGORIES.get(tool_name)

    def _is_allowed(self, tool_name: str) -> bool:
        category = self._category(tool_name)
        if category is None:
            return True
        return bool(getattr(self._permissions, self._CATEGORY_FIELDS[category]))

    def _sandbox_error(
        self, request: ToolCallRequest, workspace: str | None = None
    ) -> str | None:
        if not self._sandbox.enabled:
            return None

        root = Path(workspace).expanduser().resolve() if workspace else self._sandbox.workspace.resolve()
        for key, value in self._walk_arguments(request.arguments):
            if key.lower() not in self._PATH_KEYS:
                continue
            candidate = Path(value)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (root / candidate).resolve()
            )
            if not resolved.is_relative_to(root):
                return f"filesystem path escapes configured workspace: {value}"
        return None

    def _request_for_workspace(
        self, request: ToolCallRequest, workspace: str | None
    ) -> ToolCallRequest:
        """Make relative path arguments unambiguous for the MCP subprocess."""
        if not workspace:
            return request
        root = Path(workspace).expanduser().resolve()
        return ToolCallRequest(
            tool_name=request.tool_name,
            arguments=self._resolve_path_arguments(request.arguments, root),
        )

    @classmethod
    def _resolve_path_arguments(cls, value: Any, root: Path, key: str = "") -> Any:
        if isinstance(value, dict):
            return {
                child_key: cls._resolve_path_arguments(child, root, str(child_key))
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [cls._resolve_path_arguments(child, root, key) for child in value]
        if isinstance(value, str) and key.lower() in cls._PATH_KEYS:
            candidate = Path(value).expanduser()
            return str(candidate if candidate.is_absolute() else (root / candidate).resolve())
        return value

    async def _call_in_sandbox(
        self, request: ToolCallRequest, workspace: str | None = None
    ) -> ToolCallResult | None:
        if not self._sandbox.enabled or self._category(request.tool_name) != "terminal":
            return None
        command = request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        if self._sandbox_pool is None:
            self._sandbox_pool = ContainerPool(
                image=self._sandbox.image or DEFAULT_IMAGE,
                network=False,
            )
        root = Path(workspace).expanduser().resolve() if workspace else self._sandbox.workspace.expanduser().resolve()
        try:
            runner = await self._sandbox_pool.get("langgraph", str(root))
        except Exception as exc:  # noqa: BLE001 - optional sandbox boundary
            return ToolCallResult(
                success=False,
                output=None,
                error=f"sandbox unavailable: {exc}",
            )
        if runner is None:
            reason = self._sandbox_pool.reason or f"could not mount workspace {root}"
            return ToolCallResult(
                success=False,
                output=None,
                error=f"sandbox unavailable: {reason}",
            )
        try:
            result = await runner.run(command, timeout=300)
        except Exception as exc:  # noqa: BLE001 - optional sandbox boundary
            return ToolCallResult(success=False, output=None, error=f"sandbox execution failed: {exc}")
        output = result.combined()
        if result.timed_out:
            return ToolCallResult(success=False, output=None, error="command timed out in sandbox")
        if result.exit_code != 0:
            return ToolCallResult(success=False, output=output, error=f"command failed in sandbox (exit {result.exit_code})")
        return ToolCallResult(success=True, output=output or "(no output)")

    @classmethod
    def _walk_arguments(cls, value: Any, key: str = ""):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                yield from cls._walk_arguments(child_value, str(child_key))
        elif isinstance(value, list):
            for child in value:
                yield from cls._walk_arguments(child, key)
        elif isinstance(value, str):
            yield key, value

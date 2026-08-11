# gateway-server

A **composition gateway** for the operating-agent MCP fleet. It mounts every
sub-server (filesystem, git, terminal, search) behind a single FastMCP
instance, so any MCP client reaches all tools through **one** connection
instead of opening one client per server.

Built on FastMCP 3.x [server composition](https://gofastmcp.com/servers/composition)
(`FastMCP.mount`, a live link). `import_server` is deprecated in 3.x and is not
used here.

## Namespacing

Each sub-server is mounted under a namespace, which becomes the tool-name
prefix. The namespaces match the categories in the native agent's
`tools.enabled` config, so category filtering lines up end to end:

| Namespace    | Source server     | Tools |
|--------------|-------------------|-------|
| `filesystem` | `file-server`     | `filesystem_read_file`, `filesystem_write_file`, `filesystem_delete_file`, `filesystem_copy_file`, `filesystem_move_file`, `filesystem_rename_file`, `filesystem_list_directory`, `filesystem_create_directory`, `filesystem_delete_directory`, `filesystem_exists`, `filesystem_metadata`, `filesystem_search_files`, `filesystem_watch_directory`, `filesystem_health` |
| `git`        | `git-server`      | `git_git_status`, `git_list_branches`, `git_git_log`, `git_diff`, `git_health` |
| `terminal`   | `terminal-server` | `terminal_run_command`, `terminal_list_processes`, `terminal_health` |
| `search`     | `search-server`   | `search_index_documents`, `search_search_documents`, `search_list_indices`, `search_health` |

Plus one gateway-level tool, `gateway_health`, which reports liveness and the
mounted namespaces in a single probe.

> The cosmetic doubles (`git_git_status`, `search_search_documents`) come from
> tools that already carried their domain prefix; the namespace is applied
> as-is so the sub-servers stay untouched.

## Run it

From the workspace root, after `uv sync --all-packages`:

```bash
# stdio (default transport - what MCP clients launch as a subprocess)
./.venv/Scripts/python -m gateway_server

# or via the installed script
gateway-server
```

To serve over HTTP instead, run a tiny launcher:

```python
from gateway_server import mcp

mcp.run(transport="http", host="127.0.0.1", port=8000)
```

## Use from the native agent

The native agent's `FastMCPToolProvider` builds this gateway and talks to it
over FastMCP's in-memory transport - no subprocess, no network. Nothing extra
to configure; it is the default provider.

## Use from a LangGraph agent

Point a LangGraph MCP client at the gateway command (stdio) or URL (HTTP) and
you get every namespaced tool at once. This uses
[`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters),
which is **not** bundled with this workspace - install it in your agent's
environment first:

```bash
pip install langchain-mcp-adapters langgraph
```

```python
# stdio: LangGraph launches the gateway as a subprocess
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient(
    {
        "operating-agent": {
            "command": "gateway-server",   # or: python -m gateway_server
            "args": [],
            "transport": "stdio",
        }
    }
)

tools = await client.get_tools()          # all namespaced tools, one fleet
agent = create_react_agent("openai:gpt-4o", tools)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "list files in the current directory"}]}
)
```

For HTTP, run the gateway with `transport="http"` and swap the client entry:

```python
client = MultiServerMCPClient(
    {
        "operating-agent": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable_http",
        }
    }
)
```

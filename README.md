# Autonomous AI Operating Agent

A desktop AI agent that plans and executes multi-step goals on the local machine through
sandboxed, MCP-exposed tools — implemented twice against the same tool layer: once as a
hand-written **Plan-and-Execute + ReAct** loop, and once with **LangGraph** — to produce a
measured comparison of orchestration overhead, latency, reliability, and code volume.

> Phase 1 (MVP) scope: File/Terminal/Git MCP servers, both agent tracks, sandboxed execution,
> verification, deterministic risk classification, and the evaluation harness.
> Phase 2 (stretch): semantic memory, RAG knowledge base, full tracing dashboard, UI polish.

---

## Repo layout

```
.
├── apps/
│   ├── desktop/                  # Tauri 2 shell
│   ├── frontend-native/          # React UI — native track
│   └── frontend-langgraph/       # React UI — langgraph track
├── packages/                     # uv workspace members (Python)
│   ├── api/                      # FastAPI backend
│   ├── agent-native/              # Plan-and-Execute + ReAct
│   ├── agent-langgraph/           # LangGraph StateGraph
│   ├── mcp-servers/
│   │   ├── file-server/
│   │   ├── terminal-server/
│   │   ├── git-server/
│   │   ├── search-server/
│   │   ├── knowledge-server/     # Phase 2
│   │   └── memory-server/        # Phase 2
│   ├── sandbox/                  # Docker sandbox wrapper
│   ├── observability/            # OTel + Langfuse instrumentation
│   ├── evaluation/                # Benchmark harness
│   └── common/                    # Shared types/config/utils
├── infra/                         # docker-compose for postgres / qdrant / langfuse (TODO)
├── docs/
└── pyproject.toml                 # workspace root (virtual — no code of its own)
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | latest | Python package/workspace manager |
| Python | 3.12+ | installed automatically by `uv` if missing |
| Node.js + [pnpm](https://pnpm.io/) | LTS | for `apps/*` frontends |
| Docker Desktop | latest | for sandboxed tool execution + infra (Postgres, Qdrant, Langfuse) |
| Rust toolchain | stable | required by Tauri (`apps/desktop`) |

---

## First-time setup

Everything below has already been run once for this repo — kept here so a teammate (or a
fresh clone) can reproduce it, and as a reference for adding a **new** package later.

### 1. Root workspace `pyproject.toml`

Already in place at the repo root:

```toml
[project]
name = "autonomous-ai-agent-workspace"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.uv]
package = false

[tool.uv.workspace]
members = ["packages/*", "packages/mcp-servers/*"]
exclude = ["packages/mcp-servers"]
```

### 2. Initialize each package

> `--no-workspace` is required here: until every glob-matched folder has its own
> `pyproject.toml`, `uv` can't validate the workspace, so it refuses to run `init` inside any
> single member. Skip it and you'll hit `Failed to discover parent workspace`.

```bash
uv init --package packages/api --python 3.12 --no-workspace
uv init --package packages/agent-native --python 3.12 --no-workspace
uv init --package packages/agent-langgraph --python 3.12 --no-workspace
uv init --package packages/sandbox --python 3.12 --no-workspace
uv init --package packages/observability --python 3.12 --no-workspace
uv init --package packages/evaluation --python 3.12 --no-workspace
uv init --package packages/common --python 3.12 --no-workspace

uv init --package packages/mcp-servers/file-server --python 3.12 --no-workspace
uv init --package packages/mcp-servers/terminal-server --python 3.12 --no-workspace
uv init --package packages/mcp-servers/git-server --python 3.12 --no-workspace
uv init --package packages/mcp-servers/search-server --python 3.12 --no-workspace
uv init --package packages/mcp-servers/knowledge-server --python 3.12 --no-workspace
uv init --package packages/mcp-servers/memory-server --python 3.12 --no-workspace
```

### 3. Verify every member has a `pyproject.toml`

```powershell
# PowerShell
Get-ChildItem -Recurse -Filter pyproject.toml -Path packages
```
```bash
# macOS/Linux
find packages -name pyproject.toml
```
You should see 13 results (7 top-level packages + 6 under `mcp-servers/`).

### 4. Sync the whole workspace

```bash
uv sync --all-packages
```
This is the point `uv` actually reads the root `[tool.uv.workspace]` table, resolves every
member into one `uv.lock`, and builds a single shared `.venv` at the repo root.

### 5. Frontend + desktop shell (separate from the uv workspace)

```bash
cd apps/frontend-native      && pnpm install
cd ../frontend-langgraph      && pnpm install
cd ../desktop                 && pnpm install
```

---

## Daily-use commands

### Adding a dependency to one package
```bash
cd packages/agent-native
uv add httpx
```

### Adding a dependency on another workspace package (internal)
```bash
cd packages/agent-native
uv add common
```
Then confirm `pyproject.toml` resolves it locally rather than from PyPI:
```toml
[tool.uv.sources]
common = { workspace = true }
```

### Re-syncing after any pyproject.toml change
```bash
uv sync --all-packages
```

### Running a package's code
```bash
# API backend
uv run --package api uvicorn api.main:app --reload

# An MCP server
uv run --package file-server python -m file_server

# Native agent track
uv run --package agent-native python -m agent_native

# LangGraph agent track
uv run --package agent-langgraph python -m agent_langgraph
```

### Running tests for one package
```bash
uv run --package agent-native pytest
```

### Adding dev tooling (lint/format/test) at the workspace level
```bash
uv add --dev ruff pytest --package api
```

### Adding a brand-new package later
```bash
uv init --package packages/<new-package> --python 3.12 --no-workspace
uv sync --all-packages
```

---

## Infra (Postgres / Qdrant / Langfuse)

Not yet scaffolded — planned under `infra/docker/`. Once added:
```bash
docker compose -f infra/docker/docker-compose.yml up -d
```

---

## Notes

- The workspace root (`package = false`) is **virtual** — it has no importable code and can't
  be published; it only exists to declare `[tool.uv.workspace]`.
- Each MCP server is a standalone `fastmcp` process, consumed identically by both agent tracks
  — this is what keeps the native-vs-LangGraph comparison isolated to the orchestration layer.
- `apps/*` are not part of the `uv` workspace — they're managed with `pnpm`/Tauri tooling.
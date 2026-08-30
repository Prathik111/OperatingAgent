# api

The HTTP / WebSocket entry point for OperatingAgent. It turns a submitted goal
into a background agent run, persists the run to the Postgres system-of-record
(or an in-memory store in dev/CI), and streams execution events back to clients
over Server-Sent Events and WebSocket.

It implements the `packages/api` collaborators from
`docs/architecture/class-diagram.mermaid` — `TaskService`, `ApprovalGateway`,
`TaskRepository`, `TaskRouter` (the `/tasks` routes) and `TaskStreamSocket`
(the streaming routes) — and dispatches to an `IAgentOrchestrator` (the
LangGraph track, plus a placeholder native track) without modifying `common`,
`agent-langgraph` or `observability`.

## Run it

```bash
uv sync
uv run api                       # uvicorn on 127.0.0.1:8080
# equivalently: uv run uvicorn api.app:create_app --factory --port 8080
```

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/tasks \
  -H 'content-type: application/json' \
  -d '{"goal":"say hi","track":"native"}'
curl http://127.0.0.1:8080/tasks/<id>
curl -N http://127.0.0.1:8080/tasks/<id>/events      # SSE stream
```

The **native** track responds with no LLM credentials configured. The
**langgraph** track boots whenever its model integration imports cleanly; a run
degrades to a `FAILED` result (never a crash) when no live model/gateway is
reachable.

## Configuration (environment)

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_HOST` | `127.0.0.1` | bind host |
| `API_PORT` | `8080` | bind port (8000 is the MCP gateway) |
| `API_LOG_LEVEL` | `info` | uvicorn log level |
| `DATABASE_URL` | _unset_ | Postgres DSN; when set the repository defaults to `postgres` |
| `API_REPOSITORY_BACKEND` | `postgres` if `DATABASE_URL` else `memory` | `memory` \| `postgres` |
| `API_CORS_ORIGINS` | `*` | comma-separated allowed origins |
| `API_DEFAULT_TRACK` | `langgraph` | `native` \| `langgraph` used when a request omits `track` |
| `API_APPROVAL_THRESHOLD` | `review` | lowest risk level that requires a human gate |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL` | `ollama` / `llama3.1` / _unset_ | model for the langgraph track |
| `AGENT_PROMPT_DIR` | `prompts` | directory holding `planner.txt` / `verifier.txt` / `responder.txt` |

Langfuse tracing follows the shared `observability` package: it is enabled only
when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present.

## Persistence

`memory` (default) is fully hermetic. `postgres` writes the 5-table run spine —
`actors → agent_threads → agent_tasks → agent_runs → agent_events`, plus the
content-addressed `config_snapshots` row each run points at. Secrets
(`llm.api_key`, `checkpoint.connection_string`) are redacted from the config
snapshot **and** from its content hash before anything is written.

## Known limitations (this pass)

- **Approvals are in-process only.** The `ApprovalGateway` classifies and gates
  tool calls in memory but is not yet wired into the orchestrators and is not
  persisted (the `approval_requests` table hangs off `plan_steps`, which is out
  of the spine scope for this change).
- Only the run spine is persisted — plans, tool calls, llm calls, verifications
  and findings are not written yet.
- Postgres repository behaviour is covered by an opt-in live test tier gated on
  `DATABASE_URL`, not by the hermetic unit suite.
# Windows note

The psycopg async pool requires a Selector event loop on Windows. API launchers
must install `asyncio.WindowsSelectorEventLoopPolicy()` before creating the event
loop (or use a server configuration that does so); the default Proactor loop is
not supported by psycopg async connections.

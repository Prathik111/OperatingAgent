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
uv run api                       # loads .env, then starts on 127.0.0.1:8000
```

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/tasks \
  -H 'content-type: application/json' \
  -d '{"goal":"say hi","track":"native"}'
curl http://127.0.0.1:8000/tasks/<id>
curl http://127.0.0.1:8000/threads
curl http://127.0.0.1:8000/threads/<thread-id>/tasks
curl -N http://127.0.0.1:8000/tasks/<id>/events      # SSE stream
```

The **native** track responds with no LLM credentials configured. The
**langgraph** track boots whenever its model integration imports cleanly; a run
degrades to a `FAILED` result (never a crash) when no live model/gateway is
reachable.

The LangGraph track launches `gateway_server` itself over FastMCP stdio. Do not
start a separate HTTP MCP process for normal API use.

## Configuration (environment)

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_HOST` | `127.0.0.1` | bind host |
| `API_PORT` | `8000` | bind port |
| `API_LOG_LEVEL` | `info` | uvicorn log level |
| `DATABASE_URL` | _unset_ | Postgres DSN; when set the repository defaults to `postgres` |
| `API_REPOSITORY_BACKEND` | `postgres` if `DATABASE_URL` else `memory` | `memory` \| `postgres` |
| `API_CORS_ORIGINS` | Tauri production origins plus `localhost:1420` | comma-separated frontend origins allowed to call the API |
| `API_ALLOWED_HOSTS` | `127.0.0.1,localhost,testserver` | accepted HTTP `Host` values; prevents DNS rebinding against the local service |
| `API_DEFAULT_TRACK` | `langgraph` | `native` \| `langgraph` used when a request omits `track` |
| `API_APPROVAL_THRESHOLD` | `review` | lowest risk level that requires a human gate |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL` | `ollama` / `llama3.1` / _unset_ | model for the langgraph track |
| `LLM_TIMEOUT_SECONDS` / `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TOP_P` | `60` / `0` / _unset_ / `1` | model generation controls |
| `MCP_GATEWAY_COMMAND` / `MCP_GATEWAY_ARGS` | current Python / `-m gateway_server` | stdio command the LangGraph track launches for tools |
| `AGENT_PROMPT_DIR` | packaged prompts | directory holding `planner.txt` / `verifier.txt` / `responder.txt` |
| `AGENT_PLANNER_PROMPT` / `AGENT_VERIFIER_PROMPT` / `AGENT_RESPONDER_PROMPT` | _unset_ | exact prompt paths overriding `AGENT_PROMPT_DIR` |
| `AGENT_MAX_ITERATIONS` / `AGENT_EXECUTION_TIMEOUT_SECONDS` / `AGENT_RETRY_ATTEMPTS` | `20` / `300` / `2` | graph and tool execution limits |
| `AGENT_STREAM` / `AGENT_ENABLE_CHECKPOINTS` / `AGENT_ENABLE_INTERRUPTS` | `true` / `true` / `true` | execution mode and resumability controls |
| `AGENT_REQUIRE_VERIFICATION` / `AGENT_REQUIRE_HUMAN_APPROVAL` | `false` / `true` | semantic verification and approval gates |
| `AGENT_CHECKPOINT_BACKEND` / `AGENT_CHECKPOINT_NAMESPACE` | `auto` / `default` | checkpoint storage and namespace |
| `AGENT_SANDBOX_ENABLED` / `AGENT_WORKSPACE` | `true` / `./workspace` | filesystem tool confinement |
| `AGENT_PERMISSION_*` | `true` | category switches for filesystem, terminal, git, search, knowledge, and memory tools |

Langfuse tracing follows the shared `observability` package: it is enabled only
when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present.

## HTTP security

The API is designed as a loopback service launched by the Tauri application, so
it does not require an application API key. Keep `API_HOST=127.0.0.1`; do not bind
to `0.0.0.0` unless the desktop-only trust model is replaced with authentication.

Browser access is restricted to `API_CORS_ORIGINS`, and requests with an
untrusted `Host` header are rejected through `API_ALLOWED_HOSTS`. Responses also
carry no-store, MIME-sniffing, framing, referrer, and permissions-policy headers.
Set the exact Tauri/Vite origin in `API_CORS_ORIGINS` rather than using `*`.

## Threads

`GET /threads` returns conversation summaries ordered by most recent activity.
`GET /threads/{thread_id}/tasks` returns that thread's tasks with their latest
run status, newest first. Both endpoints accept `limit` (1-500, default 100) and
`offset` (default 0) query parameters.

## Persistence

`memory` (default) is fully hermetic. `postgres` writes the 5-table run spine —
`actors → agent_threads → agent_tasks → agent_runs → agent_events`, plus the
content-addressed `config_snapshots` row each run points at. Secrets
(`llm.api_key`, `checkpoint.connection_string`) are redacted from the config
snapshot **and** from its content hash before anything is written.

## Known limitations (this pass)

- **Approvals are in-process only.** The LangGraph executor waits on the same
  `ApprovalGateway` exposed by `/approvals`, but pending decisions are not
  persisted and are lost on process restart.
- Only the run spine is persisted — plans, tool calls, llm calls, verifications
  and findings are not written yet.
- Postgres repository behaviour is covered by an opt-in live test tier gated on
  `DATABASE_URL`, not by the hermetic unit suite.
## Windows note

The psycopg async pool requires a Selector event loop on Windows. The `api`
console launcher passes that loop factory directly to Uvicorn, so `uv run api`
works with the PostgreSQL repository instead of selecting the incompatible
Proactor loop.

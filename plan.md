# Converge the API persistence layer onto the full canonical schema

> **Verification correction (2026-08-28):** `packages/agent-native` is a
> substantial implementation, while `packages/agent-langgraph` is the stub in
> this checkout. The LangGraph planner/executor paths cited below do not exist.
> The implemented convergence therefore targets the real shared seam: typed
> records in `common`, normalized persistence in both API repositories, metric
> event routing in `TaskService`, canonical schema init through compose, and
> native package/source hygiene. LangGraph node emission remains contingent on
> that track being implemented. The native six-table schema cannot be deleted
> without first mapping its sessions, messages, events, permissions, memories,
> resume cursor, and receipts into the canonical model.

## Context

The system of record is [infra/docker/postgres/schema.sql](infra/docker/postgres/schema.sql)
— **28 tables + 6 enums + the `v_run_metrics` view**. The only code that writes it is the API
repository, [packages/api/src/api/repository/](packages/api/src/api/repository/): a
`runtime_checkable` `Protocol` [TaskRepository](packages/api/src/api/repository/base.py) with two
backends — [InMemoryTaskRepository](packages/api/src/api/repository/memory.py) (the hermetic
default the unit suite runs against) and
[PostgresTaskRepository](packages/api/src/api/repository/postgres.py) (the system of record).

That repository writes **6 of the 28 tables** — the run spine
`actors → agent_threads → agent_tasks → config_snapshots → agent_runs → agent_events`
([_sql.py](packages/api/src/api/repository/_sql.py)). Verified: the only `INSERT INTO` targets in
all of `packages/` are those six, and there is **zero `CREATE TABLE` in `packages/`** — no rival
DDL exists anywhere. The remaining **22 tables have no writer.**

The single most visible symptom is that **`v_run_metrics` is dishonest.** The view
([schema.sql:625-634](infra/docker/postgres/schema.sql#L625-L634)) derives `steps_taken`,
`tool_calls`, `llm_calls`, `total_tokens` and `cost` from `plan_steps` / `tool_calls` /
`llm_calls` — none of which any code writes — so it returns **zero for every run**. The LangGraph
orchestrator confirms this is by design:
[`LangGraphAgent._result`](packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L214-L237)
hardcodes `llm_calls=0, total_tokens=0, cost=0.0` with the comment *"Langfuse is the source of
truth for those."* Run metrics therefore live only in traces; the database cannot answer
"how many tokens did this run cost."

**Convergence here means neither "retire a rival schema" (there is none) nor "extend the schema"
(all 28 tables and 6 enums already exist, and `common`'s enums already match them). It means: add
writers for the tables that make `v_run_metrics` honest, and stop deferring run metrics to Langfuse
alone.**

```
             HTTP  →  TaskService  →  TaskRepository (Protocol)
                            │              ├── InMemoryTaskRepository  (hermetic twin)
                            │              └── PostgresTaskRepository → schema.sql
                            │                        spine (written today):
        orchestrator.run(task, on_event) ──┐        actors, agent_threads, agent_tasks,
                            │               │        config_snapshots, agent_runs, agent_events
                    LangGraphAgent          │        metrics (this plan adds):
             (has per-call detail,          └──────► mcp_servers, tools, tool_calls, llm_calls
              discards it → Langfuse)
```

## Why this is a full rewrite

The previous `plan.md` described a different repository: `packages/agent-native` owning a
`PostgresDatabase.apply_schema()` and a 6-table native `schema.sql`, a "23-table" canonical schema,
an *unimplemented* `agent-langgraph`/`api`, and a real `sandbox` `ContainerPool`. **None of that is
true here.** `agent-native` and `evaluation` are empty stubs (`__init__.py` only); `agent-langgraph`
and `api` are fully built; `sandbox` is a stub; the schema has 28 tables; and `common`'s enums
already equal the schema's. This rewrite keeps the original **intent** — one honest canonical
database where every persisted entity points at its canonical table — and re-aims it at the code
that exists.

## Decisions taken

| Question | Decision |
|---|---|
| Schema application | Add [infra/docker/docker-compose.yml](infra/docker/docker-compose.yml) with a `postgres` service mounting `schema.sql` into `/docker-entrypoint-initdb.d/`. Nothing applies the schema today and **no compose file exists** (glob confirmed). The factory opens the pool but must not apply DDL — `initdb` owns that. |
| Retire native / eval DDL | Nothing to retire. Both packages are `__init__.py`-only stubs; there is no rival DDL. |
| Enum changes | **None.** `common.RunStatus` == the `run_status` type member-for-member, and `TaskStatus`/`AgentTrack`/`RiskLevel`/`WorkflowPhase`/`VerificationResult` all match their schema types. |
| `agent_events` reshape | **None.** Events are run-scoped (`run_id NOT NULL`, `UNIQUE(run_id, sequence_number)`) and the repo already writes them that way ([_sql.py:68-72](packages/api/src/api/repository/_sql.py#L68-L72)). The canonical table matches as-is. |
| How per-call detail reaches the repo | Reuse the existing `on_event` seam. `IAgentOrchestrator.run(task, on_event=…)` is unchanged; the orchestrator emits `llm_call` / `tool_call` typed events, and `TaskService.on_event` branches to the matching repository method. *(Alternative — a second metrics-sink callback — rejected: it changes `IAgentOrchestrator` and every call site for no gain.)* |
| Where the record contract lives | `common`. `AgentEvent`/`AgentRunResult` already live there and both `api` and `agent-langgraph` depend on it. Add `LLMCallRecord`/`ToolCallRecord` dataclasses in `common`; the event payload is the record's fields. |
| Track scope | The **LangGraph track** only. The native track is a stub ([orchestration/native_stub.py](packages/api/src/api/orchestration/native_stub.py)) and emits nothing to persist yet. |

## Gap analysis — 28 tables, 6 written

- **Written today (spine, 6):** `actors`, `agent_threads`, `agent_tasks`, `config_snapshots`,
  `agent_runs`, `agent_events`.
- **This plan adds writers (metrics, 4):** `mcp_servers`, `tools`, `tool_calls`, `llm_calls`.
  These are exactly the tables `v_run_metrics` reads.
- **Deferred (18), each named with the seam that will fill it** — see *Out of scope*.

Note the schema places metrics in child tables *on purpose*: `agent_runs` has **no** token/cost
columns ([schema.sql:134-151](infra/docker/postgres/schema.sql#L134-L151)); `llm_calls.total_tokens`
and every `duration_ms` are `GENERATED`. So the only correct way to make the view honest is to write
`llm_calls`/`tool_calls` rows — not to add columns.

---

## Phase 0 — Apply the schema

Add [infra/docker/docker-compose.yml](infra/docker/docker-compose.yml) (`postgres:16`) mounting the
canonical DDL read-only into initdb, so `docker compose up -d postgres` yields a ready database with
no Python:

```yaml
services:
  postgres:
    image: postgres:16
    environment: { POSTGRES_DB: operating_agent, POSTGRES_PASSWORD: postgres }
    ports: ["5432:5432"]
    volumes:
      - ./postgres/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
```

No bootstrap in [factory.py](packages/api/src/api/repository/factory.py): it opens the pool in the
lifespan ([app.py:40-43](packages/api/src/api/app.py#L40-L43)) and must not apply DDL.

## Phase 1 — Grow the repository contract + the hermetic twin

- [common/events.py](packages/common/src/common/events.py): the `AgentEvent` hierarchy already has
  `ToolStarted`/`ToolFinished` but they carry no structured data. Define the persistence contract as
  `LLMCallRecord` / `ToolCallRecord` dataclasses in `common`; carry a record's fields in the event
  `payload` under `type="llm_call"` / `type="tool_call"`.
- [base.py](packages/api/src/api/repository/base.py): add to the `Protocol` —
  `save_llm_call(run_id, record)`, `save_tool_call(run_id, record)`,
  `upsert_tool(server_name, base_url, tool_spec) -> tool_id`.
- [memory.py](packages/api/src/api/repository/memory.py): implement all three on `_Run` in dicts,
  plus `llm_calls_for` / `tool_calls_for` introspection helpers mirroring
  [`events_for`](packages/api/src/api/repository/memory.py#L86-L87). **Both** backends must implement
  the new methods — `TaskRepository` is `@runtime_checkable`, so a twin missing a method fails
  `isinstance`, and the hermetic unit suite runs on the twin.

## Phase 2 — Emit per-call detail from the orchestrator

The data already exists at the point of the call and is currently discarded:

- **Tool calls:** [executor.py `_invoke_once`](packages/agent-langgraph/src/agent_langgraph/nodes/executor.py#L175-L211)
  has `tool_name`, `arguments`, the classified `RiskLevel` (`_needs_approval`), `success`/`error`,
  `output` and `attempt`. Emit a `tool_call` event after each call resolves (both the success and the
  business-failure path).
- **LLM calls:** `ModelProvider` returns usage per generation; the planner/verifier/responder/executor
  nodes make those calls. Emit an `llm_call` event carrying `provider`, `model`, `prompt_tokens`,
  `completion_tokens`, `cost`, `node_name`. This is the substantive change — today
  [`_result`](packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L214-L237)
  reads none of it.
- **Mechanism:** add an emit hook to `AgentContext` (it already injects the tracer, tool registry and
  risk classifier — `langgraph_agent._build_context`), so a node can emit at the moment of the call.
  *Fallback:* accumulate records into `AgentState` and flush them from `LangGraphAgent.run` after each
  `astream` tick. Recommend the `AgentContext` hook.
- **Honest constraint to record in a comment:** MCP tool calls bypass LangChain
  ([executor.py:180-181](packages/agent-langgraph/src/agent_langgraph/nodes/executor.py#L180-L181)),
  which is *why* DB-side `llm_calls`/`tool_calls` are needed rather than "just read Langfuse."

## Phase 3 — Populate `mcp_servers` + `tools`

`tool_calls.tool_id` is a **NOT NULL** FK → `tools`
([schema.sql:271](infra/docker/postgres/schema.sql#L271)), so `tools` must exist before any tool-call
row. The [ToolRegistry](packages/agent-langgraph/src/agent_langgraph/runtime/tool_registry.py) /
[MCPAdapter](packages/agent-langgraph/src/agent_langgraph/mcp_adapter.py) discover tools from the
gateway. At discovery, upsert `mcp_servers` (`name='gateway'`, `base_url=MCP_GATEWAY_URL`) then `tools`
(`name`, `description`, `input_schema`) per spec, keyed on `UNIQUE (server_id, name)`; cache
`name → tool_id`. `save_tool_call` resolves `tool_id` from the cache (upsert-on-miss).

## Phase 4 — Write `llm_calls` / `tool_calls` in Postgres

- [_sql.py](packages/api/src/api/repository/_sql.py): add `UPSERT_MCP_SERVER`, `UPSERT_TOOL`,
  `INSERT_LLM_CALL` (`run_id, node_name, provider, model, prompt_tokens, completion_tokens, cost,
  started_at, finished_at` — `total_tokens`/`duration_ms` are generated), and `INSERT_TOOL_CALL`
  (`run_id, tool_id, arguments, success, output, error, risk_level::risk_level, risk_reason, attempt,
  started_at, finished_at`).
- [postgres.py](packages/api/src/api/repository/postgres.py): implement the three methods, wrapping the
  tool-resolve + tool-call insert in a `conn.transaction()` exactly as `save_task`/`create_run` already
  do.

## Phase 5 — Stop the orchestrator lying about metrics

With the child rows written, `v_run_metrics` is the single source of truth. Populate
`AgentRunResult`'s counters (`llm_calls`, `tool_calls`, `total_tokens`, `cost`) from the emitted
records instead of the hardcoded zeros, so the `202 → GET` path needs no join, and treat
`v_run_metrics` as the reconciliation view. Delete the *"left at zero … Langfuse is the source of
truth"* shortcut in
[`_result`](packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L214-L237).

## Phase 6 — Tests

- **Unit (hermetic, memory twin):** the three new methods persist and read back;
  `TaskService.on_event` routes `llm_call`/`tool_call` events to `save_*` while still streaming
  progress to the broker; the existing api unit tests
  ([test_routers_tasks.py](tests/unit/api/test_routers_tasks.py),
  [test_stream_sse.py](tests/unit/api/test_stream_sse.py),
  [test_routers_approvals.py](tests/unit/api/test_routers_approvals.py)) pass **unchanged** — proof the
  Protocol grew without leaking above the seam.
- **Live / integration (opt-in, gated on `DATABASE_URL`,** the tier `postgres.py` already documents):
  compose up → schema applied by initdb → run the spine plus a couple of llm/tool calls → assert
  `SELECT * FROM v_run_metrics` matches the emitted counts. This is the only test that exercises real
  SQL.

## Phase 7 — Diagrams

- [database-design.mermaid](docs/architecture/database-design.mermaid): **no new tables** (all 28
  exist). Annotate persistence edges with which class writes which table, and mark the 18 deferred
  tables as *unwritten*.
- [class-diagram.mermaid](docs/architecture/class-diagram.mermaid): correct the `api` section against
  the real code — `create_app` + lifespan ([app.py](packages/api/src/api/app.py)), routers
  `{health, tasks, stream, approvals}`, services `{TaskService, EventBroker, ApprovalGateway}`,
  repository `{TaskRepository (Protocol), InMemoryTaskRepository, PostgresTaskRepository}`, and the
  orchestration factory. *(The previous plan's "asgi.py / Router / Response" correction was itself
  wrong — no `asgi.py` exists.)*

## Verification

1. **Schema applies clean.** `docker compose -f infra/docker/docker-compose.yml up -d postgres`, then
   `psql "$DSN" -f infra/docker/postgres/schema.sql` against an empty DB — no errors.
2. **Offline gates**, per package: `uv run ruff check`, `uv run mypy`, `uv run pytest` for `api`,
   `agent-langgraph`, `common`.
3. **Metrics honest end-to-end.** Run one real LangGraph task against the DB, then
   `SELECT * FROM v_run_metrics;` — non-zero `llm_calls` / `tool_calls` / `total_tokens` / `cost` that
   match the run's receipt.
4. **No rival schema, no zero-metrics shortcut.** `rg "INSERT INTO" packages/` now includes
   `tool_calls`/`llm_calls`/`tools`/`mcp_servers`; `rg "source of truth" langgraph_agent.py` is gone;
   `rg "CREATE TABLE" packages/` is still empty.

## Out of scope, named with the seam that will receive it

- **`sandbox_sessions`** — the `packages/sandbox` effort (its own plan). `run_id`-keyed
  ([schema.sql:248-258](infra/docker/postgres/schema.sql#L248-L258)).
- **`approval_requests` + `risk_policies`** — the [ApprovalGateway](packages/api/src/api/services/approval_gateway.py)
  resolves in-memory today; persisting the decision + the matched ruleset is the natural next increment.
- **`run_phases` / `plans` / `plan_steps` / `run_findings` / `verification_results`** — LangGraph graph
  state ([graph/state.py](packages/agent-langgraph/src/agent_langgraph/graph/state.py)). `plan_steps` is
  the one borderline include (`v_run_metrics.steps_taken`), addable in Phase 4 if cheap.
- **`trace_refs`** — `AgentRunResult.metadata` already carries `langfuse_trace_id`
  ([langgraph_agent.py:224-226](packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L224-L226));
  linking it is a one-row write once wanted.
- **`task_attachments`** — no attachment ingress exists yet.
- **`evaluation_suites` / `evaluation_cases` / `evaluation_runs` / `evaluation_results` /
  `evaluation_scores`** — [packages/evaluation](packages/evaluation/) is an `__init__.py`-only stub;
  adopting the canonical eval model is its own build.
- **`memory_items` / `knowledge_sources` / `knowledge_chunks`** — the memory / knowledge MCP servers.
- **LangGraph `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` / `checkpoint_migrations`** —
  created and migrated by `PostgresSaver`
  ([checkpoint_factory.py](packages/agent-langgraph/src/agent_langgraph/checkpoint_factory.py)), never
  hand-authored, no FKs.

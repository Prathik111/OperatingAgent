# agent-native

Self-contained Plan-and-Execute + ReAct agent (v0.3.0). Implements its own
types, events, and protocols — **no imports from `packages/common`** — plus a
task repository, a tool-calling reliability harness, and a local CLI.

```
uv run --package agent-native python -m agent-native "build the project"
uv run --package agent-native agent-native-harness --provider groq --iterations 30
```

## Architecture

```
NativeAgent (orchestrator)
 ├─ Planner        structured tool-call plan (create_plan), 2 attempts
 ├─ ReactExecutor  per-step ReAct loop: risk gate → approval → MCP call → verify
 ├─ Reflector      bounded replanning (max_replans=3)
 ├─ Verifier       explicit PASS / FAIL / UNVERIFIABLE
 ├─ RiskClassifier session-tracking rules R1 (exfil) / R2 (destroy-then-publish) / R3 (repeat offender)
 ├─ ApprovalGateway  timeout auto-deny (default 120s)
 ├─ ContextCompactor  budget check + atomic pair compaction
 ├─ MCPClient      stdio JSON-RPC 2.0 client (own implementation)
 ├─ SandboxManager docker lifecycle, deny-all egress (--network none)
 ├─ TaskRepository InMemory (default) | Postgres (asyncpg, auto-bootstrap)
 └─ TracingService OpenTelemetry GenAI spans (optional `tracing` extra, no-op by default)
```

## Decisions (rationales)

1. **max_replans=3** — bounded retries with an explicit
   `REPLAN_BUDGET_EXHAUSTED` event and terminal `FAILED` run; unbounded
   replanning would hide model failures behind a hang.
2. **Explicit unverifiable step type** — `ANALYSIS` steps are a first-class
   `StepKind`; the Verifier returns `UNVERIFIABLE` for them, never a silent
   `True`. A step with no tool evidence cannot claim success.
3. **Atomic pair compaction** — only complete assistant
   `tool_calls` → `tool` result pairs are compacted, never split; a paired
   result stays paired, pre-existing orphans are preserved. An invariant test
   (`tests/test_compactor.py`) locks this in.
4. **Approval timeout 120s default** — un-attended approvals auto-deny with an
   `approval_timed_out` event instead of hanging the run; overridable via
   `--approval-timeout-s` / `AGENT_NATIVE_APPROVAL_TIMEOUT_S`.
5. **Session-tracking RiskClassifier** — the classifier keeps per-task session
   history (bounded deque), enabling R3 (repeat offender) which a stateless
   classifier cannot express.
6. **Deny-all sandbox egress** — containers run with `--network none`; hosts
   are allowlisted via TOML config, overridable through
   `AGENT_NATIVE_SANDBOX_ALLOWED_HOSTS` (comma-separated).
7. **Groq-first** — Groq is a hard dependency (`groq`, model
   `llama-3.3-70b-versatile`); Ollama is an optional extra
   (`uv sync --extra ollama`).
8. **Reliability harness** — `agent-native-harness` measures parse /
   valid-arguments / correct-tool rates against the bundled file-server
   schema fixture (`tool_schemas.json`); requires `GROQ_API_KEY` or a running
   Ollama instance.
9. **Postgres repository** — `PostgresTaskRepository` with auto-bootstrapped
   schema; local infra via `infra/docker/docker-compose.yml`
   (`postgres:16-alpine`, db/user/pass `agent_native`).
10. **Local protocol + CLI only** — `NativeAgent.run(task, on_event)` and the
    `python -m agent-native` CLI are the only entry points; `packages/api` is
    intentionally untouched.

## Config

`Settings` is loaded from a TOML file and environment variables
(`AGENT_NATIVE_*`); see `config.py` for defaults. The sandbox allowlist lives
under `sandbox.allow_egress_hosts`.

## Tests

```
uv run --package agent-native python -m pytest packages/agent-native/tests
```

Two tests are environment-gated and skip automatically: the Postgres
round-trip (needs `docker compose -f infra/docker/docker-compose.yml up -d`)
and the Docker sandbox lifecycle (needs a running Docker daemon).

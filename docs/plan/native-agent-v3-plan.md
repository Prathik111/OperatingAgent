# agent-native v3 plan — the final version

Written 2026-08-25 against v0.5.0, after the v1 build (steps 1–11 and 13 of
`docs/plan/native-agent-build-plan.md`) and the v2 hardening pass. This is the plan for the last
version: the one that finishes the thesis, makes the agent operable by someone other than its
author, and closes the distance to a 2026 frontier harness. The thirty steps below are the
authoritative v3 scope, confirmed with admin.

**Read this first, because it changes what v3 is.** The v1 plan implemented every finding in
`docs/review/frontier-agent-gap-analysis.md` — all of Parts 1 and 2, and the reporting half of
Part 3 — plus the four things named directly (sandbox, Postgres memory, subagents, parallel tools).
v2 then hardened it (migrations, honest provider tokens, a configurable `Limits`, offline tests,
lint/type config). So the gap analysis is essentially *spent*. Exactly one of its findings was left
unbuilt on purpose — **C1, the evaluation harness** — and that omission is the reason the thesis can
be argued but not shown. v3 therefore is **not** more of that list. It is three things the list
never covered: prove the comparison, run the thing in production, and reach 2026 parity.

Same conventions as the v1 plan. Each step says what changes, which files and symbols, how you know
it worked, and its size (S = a few hours, M = a day or two, L = several days). Steps are grouped
into five stages and ordered so nothing needs anything from a later step.

## One caveat on sourcing, inherited honestly

The frontier comparisons below are a strong prior, not a fresh citation. The 2026 feature set of the
leading harness (subagents, hooks, skills, checkpoint/rewind, plan mode, background tasks) was
refreshed against one live search this session; deeper live verification was not available (the
built-in search is unreliable for this model and the Firecrawl bridge needs a key). Every claim
about *our own* code was verified first-hand against the working tree on 2026-08-25 — package line
counts; the absence of hook/skill/checkpoint/multimodal/router/LSP vocabulary in `src/`; the shape
of the `Database` seam; `PartType`; `ModelRegistry`; `AgentConfig`; the delegate tool; the parallel
tool semaphore; and the in-memory MCP transport. Where a frontier system has moved on past what I
describe, your information wins.

---

## Out of scope, and defaults I chose for you

**LangGraph stays untouched.** `packages/agent-langgraph` is still a 2-LOC stub and it is another
contributor's track. The evaluation runner in Stage A *drives* it and *measures* it over the shared
tool layer and the shared task set — it never edits its code and never reshapes the LangGraph
checkpoint tables. The one thing that would invalidate the whole comparison is the two tracks
drifting onto different tools or different tasks, so the runner's job is to hold that contract, not
to modify either side.

**Two items are optional and flagged cheap-to-reverse**, so you can cut them without disturbing
anything else: the web UI (step 15) and semantic/vector memory (step 30). Everything else is
load-bearing for "final version."

**Two items from the earlier draft are deliberately not in this list**, recorded here so nothing is
lost silently: *streaming steering* (mid-run message injection — note the mid-stream **cancel**
half already shipped in v1, only the steering half is dropped), and the *packaging / deploy /
v1.0.0 release cut*. Say the word and either is easy to fold back in; the version bump in particular
is a natural close once Stages A–D are green.

**Defaults picked so the plan can be concrete** (say the word and I rework the step): traces export
over **OTLP** to any OpenTelemetry collector, with the existing per-run JSON kept as the offline
fallback (step 7); the API streams over **SSE** rather than websockets, because the event bus is
already a one-way numbered stream (step 12); checkpoints are **filesystem snapshots** of the working
folder, not a VCS shadow-branch (step 18); model routing is **config-driven**, not a learned router
(step 22); the test runner is delivered as an **MCP server** over the new transports from
step 28 rather than built in-process (step 27).

**What is already built, so v3 doesn't rebuild it** (verified 2026-08-25): the think-act-observe
loop with streaming, transient-failure retry, and mid-stream cancel; a turn's tools already run
**concurrently** under `asyncio.Semaphore(context.limits.max_parallel_tools)` with `asyncio.gather`
and per-tool timeouts; `WorkspacePolicy` path confinement plus git and shell roots and a trimmed
allowlist; the Docker sandbox (`packages/sandbox` is a real 411-LOC package — `container.py`,
`pool.py` — driven through `tools/sandbox.py`); token/cost/time on `RUN_FINISHED` and the CLI, from
provider-reported usage; `PostgresDatabase` and `MemoryDatabase` behind the 11-method `Database`
ABC, with a migration ledger; `AGENT.md` project instructions and a keyword `memories` table with
`remember`/`recall`; the single-helper `delegate` tool (`DelegateTool`, `tools/subagent.py`) and the
write-only `plan` tool; model-written compaction with the template as fallback; a richer first
prompt (folder listing + git branch); and the prompt-cache prefix marker. None of that is reopened
below.

---

## The thirty steps

| # | Step | Stage | Size | Adds / fixes |
|---|---|---|---|---|
| 1 | Evaluation task suite with reproducible fixtures | Evidence | L | C1 |
| 2 | Evaluation runner over `AgentService`, per track | Evidence | L | C1 |
| 3 | Scorers: deterministic checks + optional LLM-judge | Evidence | M | C1 |
| 4 | Run read-back on the `Database` ABC (`get_run`, `list_runs`) | Evidence | S | enables 5, 8 |
| 5 | Comparison + report view (native vs LangGraph) | Evidence | M | C4 |
| 6 | Reproducibility harness (seed, temp 0, stability check) | Evidence | S | thesis validity |
| 7 | OpenTelemetry OTLP exporter behind `Monitoring.shutdown` | Operate | M | finishes C2 |
| 8 | Run history + receipts surfaced | Operate | S | — |
| 9 | Budget governance: cost/token ceiling per run | Operate | M | — |
| 10 | Secrets handling + redaction | Operate | S | — |
| 11 | Resumable runs from the event cursor | Operate | M | — |
| 12 | REST + SSE API server | Interface | L | — |
| 13 | Async permission channel (`PermissionResponder`) | Interface | M | — |
| 14 | Session management: list / resume / fork / delete | Interface | S | — |
| 15 | Thin web UI (optional) | Interface | M | — |
| 16 | Hooks: lifecycle extension points | Parity | M | — |
| 17 | Skills: `SKILL.md` discovery + progressive disclosure | Parity | M | — |
| 18 | Filesystem checkpoints + rewind | Parity | M | — |
| 19 | Plan mode: read-only gate before execution | Parity | S | — |
| 20 | Multimodal input (images / documents) | Parity | M | — |
| 21 | Extended-thinking budget + reasoning-token surfacing | Parity | S | — |
| 22 | Model routing + fallback | Parity | M | — |
| 23 | Web fetch / search tool | Parity | M | — |
| 24 | Anchor-based surgical edit tool | Parity | M | — |
| 26 | Parallel subagent fan-out | Parity | M | — |
| 27 | Test runner + structured result parser | Parity | M | — |
| 28 | Remote MCP support (stdio / SSE / streamable-http) | Parity | L | — |
| 29 | Full test parity + CI | Release | M | — |
| 30 | Semantic / vector memory (optional, behind same seam) — **deferred 2026-08-27** | Release | M | — |
| 31 | Docs + architecture refresh | Release | S | — |

Thirty rows — step 25 (LSP integration) was cut from scope on 2026-08-27, with the later numbers
left unchanged so 26–31 still name the same work; "final version" was asked to be complete
otherwise. Step 30 (semantic / vector memory) was then deferred on 2026-08-27 as well — it was
always flagged optional, and its paraphrase-recall check needs a real embedding model the offline
verification gate cannot run; keyword memory stays the default. So Stage E as actually built is
steps 29 and 31 only. The priority order is the stage order: Evidence is the binding constraint,
and everything else is asserted until it exists.

---

## Stage A — Evidence: make the comparison showable

The deferred step 12 of the v1 plan, expanded, and first because it is the only P0 the gap analysis
left open. `packages/evaluation`, `docs/comparison/` and `docs/evaluation/` are all empty
placeholders today; this stage fills them. A capability feature built before this stage exists
cannot show what it bought — so this comes before Stages C–D on purpose.

### 1. Evaluation task suite with reproducible fixtures (L)

A versioned set of tasks in `packages/evaluation`. Each task is a small record: a goal in words, a
working folder (a fixture checked into the repo or built by a setup script), the tool set it may
use, and how to tell if it passed. Fifteen to twenty-five to start, spanning the shapes the thesis
cares about: single-file reads, multi-file edits, tasks needing the shell, tasks that *should be
refused* (a write outside the folder), tasks exercising memory across two runs, and tasks a subagent
should handle. Fixtures must be deterministic — the same starting bytes every run — because a
benchmark that measures its own setup noise measures nothing.

**Files.** New: `packages/evaluation/src/evaluation/suite.py` (the task record + loader),
`packages/evaluation/tasks/`. No `src/` change.

**Verify.** Load the suite; assert every task has a reachable fixture, a callable check, and a
unique id.

### 2. Evaluation runner over `AgentService`, per track (L)

A runner that takes a suite and a track (`native` or `langgraph`) and, per task, creates a fresh
isolated session via `AgentService.create_session`, runs it to completion, and records: passed/
failed, turns, input/output/cached tokens, wall-clock, tools called, whether a permission was asked,
cost from `Model.cost_of`, retries, and any error. One task's state must never leak into the next,
which the numbered events and per-session store already make clean. Persist to Postgres (new
`evaluation_runs` + `evaluation_results` tables — the historical appendix in
`database-design.mermaid` already sketched this shape) and mirror to JSON so a run works without a
database.

**Files.** New: `packages/evaluation/src/evaluation/runner.py`; a `schema.sql` addition; a `--track`
entry point. Reuses `service.py` (`AgentService`, `config_for`) unchanged.

**Verify.** Run the suite against `native` end to end; one result row per task, all metric columns
populated, sessions isolated.

### 3. Scorers: deterministic checks + optional LLM-judge (M)

A small scorer interface with concrete checks: final file-state assertion, command exit code, output
regex, and a "correctly refused" check for the safety tasks. Deterministic scorers first, because
they reproduce. Add one optional LLM-judge scorer for open-ended goals, clearly marked
non-deterministic and off the critical path.

**Files.** New: `packages/evaluation/src/evaluation/scoring.py`. Task records name their scorer.

**Verify.** A known-good and a known-bad transcript score pass and fail; the file-state scorer
detects a missing edit.

### 4. Run read-back on the `Database` ABC (S)

Precise gap found 2026-08-25: `Database` has `save_run` but **no way to read a run back** — no
`get_run`, no `list_runs`. Receipts reach the database and are unqueryable. Add the two methods to
the ABC (`database.py`) and implement them in both `MemoryDatabase` and `PostgresDatabase`
(`postgres.py`). Small, but it unblocks steps 5 and 8.

**Verify.** Save a run, read it back by id; list a session's runs newest-first; the shared
`Database` test set covers both stores.

### 5. Comparison + report view (native vs LangGraph) (M)

Given two result sets over the same suite, render where they diverged: the pass set, per-task turns/
tokens/cost deltas, and — because events are numbered and complete — where the tool sequences
forked. Emit a Markdown/HTML report into `docs/comparison/`. This is the artifact that makes the
thesis legible rather than asserted, and it is a consumer over existing data, not new
instrumentation.

**Files.** New: `packages/evaluation/src/evaluation/compare.py`; output under `docs/comparison/`.
Reads step 4's methods and the event stream.

**Verify.** Feed two runs over the same suite; the report's pass-set difference and per-task cost
table match the raw rows.

### 6. Reproducibility harness (S)

Pin the knobs that make the comparison honest: temperature 0 (already the `AgentConfig` default), a
fixed seed where the provider supports it, and pinned fixtures. Run the suite twice and assert the
pass set is stable; an unstable suite measures noise and must be fixed before any number is quoted.

**Verify.** Two back-to-back runs produce the same pass set; a deliberately flaky task is flagged.

---

## Stage B — Operate: reliability and observability

The agent runs; it is not yet operable by someone who is not watching it.

### 7. OpenTelemetry OTLP exporter behind `Monitoring.shutdown` (M)

`Monitoring` places spans correctly (run, turn, tool) but `shutdown()` in `monitoring.py` is a no-op
with no exporter — the `tracing` extra has waited since v1. Wire a real OTLP exporter so spans reach
any collector, keeping the per-run JSON as the offline fallback. This finishes C2: the points exist,
the sink does not, and every interesting native-vs-LangGraph number is cross-run.

**Verify.** Point at a local collector, run once, see one trace per run with a span per turn and
tool; with no collector, the JSON still lands.

### 8. Run history + receipts surfaced (S)

Using step 4's read-back, add a `runs` view (a CLI subcommand, later an API route) listing recent
runs for a session or folder with turns, tokens, cost and duration. The numbers are stored already;
this is the surface that turns "did it get cheaper" into a query.

**Verify.** After several runs, the view lists them with totals matching the `runs` rows.

### 9. Budget governance: cost/token ceiling per run (M)

`Limits` already carries per-run knobs (turns, wall-clock, retries, `max_parallel_tools`). Add a
token and a cost ceiling that stop a run *cleanly* with a partial result and a clear reason, rather
than letting a runaway loop bill without bound. Coordinate backoff across parallel tools and
subagents so a 429 isn't retried N-ways at once.

**Files.** `loop.py` (`Limits`, the turn loop), `service.py` where limits are threaded.

**Verify.** A run with a tiny ceiling stops at it, reports it as the stop reason, and still returns
what it had.

### 10. Secrets handling + redaction (S)

The only secret today is `GROQ_API_KEY` via `load_dotenv`. Before anything writes runs to a shared
store or ships them over an API, add a redaction pass so keys and obvious secrets never land in
events, traces, JSON exports, logs, or memory rows. A pluggable secret source (env by default) keeps
the key out of code paths that serialize.

**Verify.** A run whose text contains a key-shaped string exports masked; grep the JSON and event
rows to confirm nothing leaks.

### 11. Resumable runs from the event cursor (M)

Events are numbered and `load_events(after_sequence=...)` already replays from a cursor — the read
path exists. Add the *run* path: after a crash or kill, reattach to a session and continue from the
last event rather than starting over. Postgres makes this durable across processes. Matters most for
long unattended eval runs (Stage A) and for the API (step 12), where a dropped connection should not
lose work.

**Verify.** Kill a run mid-flight, resume, and confirm it continues from the last event with no
duplicate side effects.

---

## Stage C — Interface: make it usable by someone else

The CLI is no longer the only entry point for Step 12 — `packages/api` now ships a FastAPI application, orchestration, repositories and services integrating the native `AgentRuntime` and Postgres/Memory persistence for REST + SSE. Steps 13–15 remain future work.

### 12. REST + SSE API server (L) — **delivered** (Stage C Step 12 only)

Shipped as `packages/api/src/api/{app.py,config.py,repository/{memory,postgres,factory},orchestration/factory.py,services/{task_service,event_broker,approval_gateway}}` plus `native/{runtime.py,routers/sessions,messages,events,permissions,runs,health}` integrating `AgentRuntime`/`AgentService` and native Postgres/Memory persistence. Create a session, post a message and stream events over SSE, replay from cursor, list/get runs, manage permissions, and expose native health.

Remaining gaps (not a stub): native WebSocket parity (`/native/ws/*`), API auth/rate-limiting, production Postgres migration enablement, and the broader Stage C surface — session fork/delete management and the thin web UI (Steps 13–15) — which are future work.

**Files.** Delivered under `packages/api/src/api/`; consumes `AgentService` and the event bus.

**Verify.** Drive a full task over HTTP — create, stream to completion, replay by cursor, read the receipt — covered on both Task and native tracks.

### 13. Async permission channel (`PermissionResponder`) (M)

Approvals are answered on CLI stdin today. Extract a `PermissionResponder` interface with two
implementations: the existing terminal prompt, and an async one that surfaces the request over the
API and waits for a decision. The permission *model* (`WorkspacePolicy`, `RulePolicy`,
`PermissionGrant.argument_pattern`) is untouched — only where the answer comes from changes.

**Verify.** Over the API, a tool needing approval pauses, the client answers, the run proceeds;
denial is reported cleanly.

### 14. Session management: list / resume / fork / delete (S)

Commands (CLI and API) to list sessions with receipts, resume one, fork one (branch a conversation
to try an alternative), and delete. Everything is keyed by session id already; this is the
management surface over it.

**Verify.** Fork a session, diverge the two, confirm each has its own independent event stream.

### 15. Thin web UI (optional) (M)

*Optional, cheap to cut.* A single-page view over the API: the conversation, the live event stream,
approve/deny buttons, per-run receipts. Nothing the API doesn't already expose.

**Verify.** Watch a task run live in the browser, approve a tool, read the receipt.

---

## Stage D — Frontier parity 2026

The loop's *shape* is already frontier-grade. This stage is capability breadth. Everything here was
confirmed absent from `src/` on 2026-08-25. It is the largest stage; within it, one dependency is
worth stating up front: **step 28 (remote MCP transports) is the enabler for step 27** — the test
runner is cleanest as an external MCP server, so if you build it on top of 28, do 28
first.

### 16. Hooks: lifecycle extension points (M)

User-configured callbacks at defined moments — before a tool runs, after it returns, when a prompt
is submitted, when a run or subagent stops. The event bus (`events.py`) is the substrate; hooks are
dispatch on top of it, able to observe and (at the pre-tool point) veto. This is how a user adds
auto-formatting, logging, or a policy gate without editing the agent.

**Files.** New hook registry + dispatch, called from `loop.py` (tool run points) and `service.py`
(prompt/stop points).

**Verify.** A pre-tool hook that blocks a command is honored; a post-tool hook fires with the
result; disabling hooks restores current behavior exactly.

### 17. Skills: `SKILL.md` discovery + progressive disclosure (M)

`AGENT.md` is loaded, but there is no notion of a *skill* — a named folder of instructions and
resources pulled in only when relevant. Discover `SKILL.md` files from the workspace, list their
names and one-line descriptions cheaply in the prompt, and inject a skill's full body only when the
model invokes it. This is the progressive-disclosure pattern the 2026 harness ships, and it fits the
"operating agent" framing directly.

**Files.** Discovery in `config.py`/`PromptBuilder` (which already assembles `project_instructions`
and `remembered`); a small invoke-skill tool.

**Verify.** A skill's body is absent from the base prompt and present after the model invokes it;
behavior changes accordingly.

### 18. Filesystem checkpoints + rewind (M)

The numbered event stream replays the *conversation*; nothing snapshots the *files*. Add a checkpoint
before a batch of edits and a rewind that restores the working folder to a prior checkpoint — the
safety net for a wrong or destructive edit. Default to filesystem snapshots of the working folder (a
VCS shadow-branch is the heavier alternative, cheap to swap to). Pairs naturally with the
anchor-edit tool (24).

**Files.** New checkpoint module; hook points around the write/edit tools via step 16 or
`ToolManager`.

**Verify.** Make edits, checkpoint, make more, rewind, confirm the folder matches byte-for-byte.

### 19. Plan mode: read-only gate before execution (S)

Distinct from the write-only `plan` tool (which lets the model *record* a list). Plan mode is a run
*mode* that restricts the agent to read-only tools, has it produce an approach, and requires approval
before it may switch to mutating tools. The confinement machinery (`allowed_tools` on `AgentConfig`,
the permission floor) exists already; this composes them into a gate.

**Verify.** In plan mode the agent cannot write or run a mutating command until the plan is approved;
approval flips it to full tools.

### 20. Multimodal input (images / documents) (M)

`PartType` today is `TEXT`, `REASONING`, `TOOL_CALL`, `COMPACTION` — no media. Add a media
`MessagePart` so images and documents can enter a `Conversation`, and teach the provider adapters
(`groq_model.py`, `ollama_model.py`) to encode them for vision-capable models. Gate on provider
capability so a text-only model degrades cleanly.

**Verify.** Send an image to a vision-capable model and get a grounded answer; a text-only model
reports the input unsupported rather than crashing.

### 21. Extended-thinking budget + reasoning-token surfacing (S)

The substrate exists: `StreamType.REASONING`, a `Reasoning` part, and Ollama already emits thinking
chunks. What's missing is control and accounting. Add a reasoning-effort / thinking-budget knob on
`Limits` (and `AgentConfig`), pass it to providers that support it, and surface reasoning tokens in
the receipt so their cost is visible.

**Verify.** Raising the budget increases reasoning tokens in the receipt; a provider without the
feature ignores it without error.

### 22. Model routing + fallback (M)

`ModelRegistry` (`register_model`/`register_provider`/`get`/`get_provider`) is the seam, and
`Subagent.model` already lets a helper use a different model — but the main loop uses a single
`AgentConfig.model` with no fallback. Add config-driven routing: a strong model for the main loop, a
cheaper one for subagents and simple turns, and automatic fallback to an alternate provider on
failure or rate-limit (composing with the retry that already exists). Config-driven, not a learned
router (cheap to make smarter later).

**Verify.** Kill the primary provider mid-run and watch the fallback take over; subagents bill
against the cheaper model in the receipt.

### 23. Web fetch / search tool (M)

`builtins.py` is empty by design — every tool comes from MCP — so this arrives as an MCP server (a
natural fit for the remote transports in step 28). A gated, sandboxed fetch/search tool, because a
2026 operating agent that cannot read a page is sharply limited. Keep the approval prompt, and route
it through the sandbox so network reach is deliberate.

**Verify.** The tool is refused without approval and confined when sandboxed; a fetch returns
readable content the model can act on.

### 24. Anchor-based surgical edit tool (M)

Alongside whole-file write, an anchor/patch-based edit tool so large files are changed surgically
rather than rewritten — cheaper, less destructive, and a clean pair with checkpoints (18). Arrives
as an MCP tool like the file server.

**Verify.** The tool changes a target region of a large file without touching the rest; a
non-matching anchor fails safely rather than editing the wrong place.

### 26. Parallel subagent fan-out (M)

Today `DelegateTool` (`tools/subagent.py`) runs one helper per call. Because a turn's tool calls
*already* run concurrently under `asyncio.Semaphore(context.limits.max_parallel_tools)` with
`asyncio.gather` (v1 step 10), a model that issues several `delegate` calls in one turn already
fans out — so the missing piece is not raw concurrency but making it first-class and safe: a
dedicated fan-out that maps one helper over a list of inputs and gathers the results, plus verifying
nested re-entrancy (a helper's own loop under the parent's semaphore), a shared cancel that stops all
children, and event tagging that keeps each child's stream attributable. Respect `helper_max_turns`
per child so a fan-out can't multiply into a runaway bill.

**Files.** `tools/subagent.py` (`DelegateTool`, `MAX_HELPER_TURNS`, `HELPER_RUN_SEPARATOR`),
`loop.py` (the semaphore/gather path and `Limits`).

**Verify.** Fan a helper across five inputs; the turn costs about the slowest child, not the sum;
cancelling the parent stops all five; each child's events are attributable.

### 27. Test runner + structured result parser (M)

No test-runner tool exists; the only shell is the gated `terminal_run_command`. Add a tool that runs
the project's test command and parses the output into structure the model can act on — pass/fail
counts, the names of failing tests, and the first failure's message — rather than handing back a wall
of text. Delivered as an MCP server (again, a fit for step 28), so it composes with the sandbox and
the permission floor.

**Verify.** On a project with one failing test, the tool returns a structured result naming the
failing test and its message; the agent uses it to locate the fault.

### 28. Remote MCP support (stdio / SSE / streamable-http) (L)

Today the bridge uses FastMCP's **in-memory transport** — `connect()` builds the file/git/terminal
gateway in-process and wraps it in `Client(gateway)`, with "no subprocess and no socket"
(`tools/mcp_bridge.py`). Add real transports so the agent can reach *external* MCP servers over
stdio, SSE, and streamable-http: a configured server list, connection lifecycle and auth, and
health/timeout handling, keeping the in-memory gateway as the default for the local file/git/terminal
tools. This is the foundation the richer tools ride on — the web tool (23) and the test runner (27)
are naturally external servers once this exists.

**Files.** `tools/mcp_bridge.py` (`MCPToolProvider.connect`, the `Client` construction, `build_gateway`).

**Verify.** Connect to an external MCP server over each transport, list and call one of its tools,
and confirm a dead server fails cleanly without taking the run down; the local in-memory tools still
work unchanged.

---

## Stage E — Release: harden and document

### 29. Full test parity + CI (M)

The offline stdlib runner is a subset (it skips tests needing `groq`, `asyncpg`, `fastmcp`, `mmdc`).
Get the full `pytest` suite runnable where the extras are installed, add tests for the eval runner
(Stage A), the API (12), hooks (16), checkpoints (18), routing (22), multimodal (20), fan-out (26)
and remote MCP (28), and wire CI to run `ruff`, `mypy` (both configured in v2) and the tests on
every change.

**Verify.** CI is green on a clean checkout with extras installed; the eval suite runs in CI against
a fake provider.

### 30. Semantic / vector memory (optional, behind same seam) (M) — DEFERRED

**Deferred 2026-08-27.** Not built. It was always flagged optional and cheap-to-reverse, and its
verify step (below) needs a real embedding model that the offline verification gate cannot run, so
it sits outside the "verify offline at each step" contract the rest of v3 was held to. Keyword
memory stays the default; the seam described below is intact, so this remains a clean future swap-in
rather than a rewrite. The design note is kept as-is for whoever picks it up.

*Optional, cheap to cut.* v1 chose keyword memory and no embeddings. If semantic recall is wanted,
add an embedding-backed store *behind the existing `Database`/memory seam* so nothing above it
changes — the historical `database-design.mermaid` appendix even sketched the vector shape
(`MEMORY_POINTS`). Keyword memory stays the default; this is a swap-in, not a rewrite.

**Verify.** With semantic memory on, a paraphrased query recalls a note keyword matching would miss;
with it off, behavior is exactly as today.

### 31. Docs + architecture refresh (S)

Bring the docs to v3: a `native-agent-v3.md` architecture note (successor to `native-agent-v2.md`),
an API reference for Stage C, a "how to reproduce the thesis comparison" runbook pointing at the eval
runner and the report, and an update to `database-design.mermaid`'s live section for the new eval
tables (keeping the v0.3 appendix as-is). Fill the `docs/evaluation/` placeholder with the suite's
description.

**Verify.** A reader following the runbook cold can reproduce a comparison report; every file/symbol
the docs cite resolves.

---

## Order, if you're doing this alone

Stage A first and entire — **the ordering I would defend hardest.** Nothing else in v3 can be shown
to have helped until there is something to measure with; the thesis's binding constraint is
evidence, not capability. Build 1–3, then 4 (small, unblocks the rest), then 5 and 6 so the
comparison is both legible and trustworthy.

Then Stage B, because an unattended benchmark needs the budget ceiling (9), trace export (7) and
resumability (11) to be trusted — and secrets redaction (10) must precede any shared store or API.
Stage C next, since the API (12) leans on resumable runs and gives the permission channel (13) and
session tools (14) a home; the UI (15) is optional and last in the stage.

Stage D is the big one. Slot the cheap high-value items early — extended-thinking surfacing (21) is
S, plan mode (19) is S. Do **remote MCP (28) before the web tool (23) and the test runner
(27)**, since those are cleanest as external servers on top of it. Model routing (22) wants Stage A
in place to show the cost win — the same lesson as v1's "step 3 before step 8." Multimodal (20),
skills (17), hooks (16), checkpoints (18) and fan-out (26) are independent of each other and can go
in any order once their small substrate notes above are respected.

Stage E closes it: parity tests and CI (29) and the docs refresh (31). The optional semantic-memory
swap (30) was deferred on 2026-08-27 (see its note above), so the stage as built is 29 and 31.

If two people are working, Stage A and Stage B's non-dependent pieces (7, 10) can start on day one;
the API (12) and the frontier features (Stage D) should not begin until Stage A can measure them, or
you lose the before-and-after that justified building them.

---

Every package line count, symbol name, `PartType` and `Database` method, transport detail, and
"absent from `src/`" claim above was checked against the working tree at v0.5.0 on 2026-08-25. As in
the v1 plan, line numbers were left out because they drift the moment step 1 lands — the file and
symbol names are the durable anchors.

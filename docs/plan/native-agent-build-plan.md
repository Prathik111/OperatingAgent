# agent-native build plan

Everything from `docs/review/frontier-agent-gap-analysis.md`, in the order I'd build it, plus the
four pieces admin named directly: a sandbox, memory in Postgres, subagents, and parallel tool
calls. Written 2026-08-23 against v0.5.0.

Twelve steps in five stages. Each step says what changes, which files, how you know it worked, and
roughly how big it is (S = a few hours, M = a day or two, L = several days). Steps are in dependency
order — nothing here needs anything from a later step.

**Out of scope.** The LangGraph agent is another contributor's work, so nothing below touches
`packages/agent-langgraph` or the LangGraph checkpoint tables. Two other things I'm deliberately
leaving out to keep this small: no vector database and no embeddings (memory is plain rows you can
read with SQL), and not the 20-table design in `docs/architecture/database-design.mermaid` — that
diagram describes the old v0.3 shape (plan steps, verification results, Qdrant) and doesn't match
the domain model the code actually has. Step 6 builds six tables that match today's `Database`
interface instead. The old diagram should be marked historical rather than followed.

**Two choices I made for you, both cheap to reverse.** The sandbox is Docker, because that's what
`ExecutionMode.SANDBOX` already promises ("a locked-down container", `tools/base.py:28`) and what
the old v0.3 code used — say so if you'd rather have something lighter and I'll rework step 11.
And memory is keyword-matched rows rather than semantic search, which is the simple version; the
richer one is a later pass and doesn't change any of the interfaces below.

---

## The twelve steps

| # | Step | Stage | Size | Fixes |
|---|---|---|---|---|
| 1 | Fix the permission classification, give git a root | Safety | S | B2 |
| 2 | Confine the shell tool | Safety | S | B3, B4 |
| 3 | Report tokens, cost and time on every run | Measure | S | C3 |
| 4 | Retry a turn when the model call fails transiently | Measure | S | A8 |
| 5 | Turn Monitoring on and let it export | Measure | M | C2 |
| 6 | Postgres store behind the existing interface | Remember | M | deferred Postgres |
| 7 | Memory that survives between runs | Remember | M | A5 |
| 8 | Subagents | Reach | M | A1 |
| 9 | A plan the model writes down | Reach | S | A2 |
| 10 | Run a turn's tools in parallel, with timeouts | Reach | S | A6 |
| 11 | Run tools in a container | Sandbox | L | B1 |
| 12 | The evaluation runner and task set | Measure | L | C1, C4 |
| 13 | Six small wins | Polish | M | A3, A4, A7, A9, B5 |

Step 12 sits at the end of the table but is written to be startable any time after step 5 — see the
note there. Thirteen rows, twelve steps, because step 13 is a bundle of small independent things.

---

## Stage 1 — Close the safety holes

Do these first. They're small, and one of them is the only place where the system is quietly
weaker than it presents.

### 1. Fix the permission classification, give git a root (S)

Two changes that go together.

In `mcp_bridge.py:_infer_permissions` (~line 214), stop treating "doesn't write" as "doesn't need
asking". Add a third flag to `ToolPermissions` — something like `reaches_paths: bool` — and set it
on any tool that takes a path argument. The policy floor in `_capability_floor`
(`permissions.py:109-120`) then asks for a path-taking tool even when it's read-only — today its
read-only branch (`permissions.py:116-117`) returns a flat ALLOW. This is what makes `git_status` on
`/somewhere/else` prompt instead of running silently. The branch that produces the wrong verdict is
`mcp_bridge.py:235`, the `git` read-only fall-through.

In `git_service.py:26-38`, add the same root confinement the file server has: read a
`GIT_SERVER_ROOT` environment variable, resolve the `repository` argument against it, and raise if
it escapes. Set it from `MCPToolProvider.connect(root=...)` next to the existing
`FILE_SERVER_ROOT` line (`mcp_bridge.py:132`).

**Verify.** A unit test that `_infer_permissions("git_status")` no longer returns a bare
`read_only=True`; a test that `GitService().status("/etc")` raises; and a live run where the agent
asks before a `git_status` outside the folder.

### 2. Confine the shell tool (S)

`terminal_service.py:127` takes its working directory from a model-supplied argument and defaults
to `None`, meaning the agent's own directory. Give the terminal server a root the same way, default
every command's working directory to it, and reject a `cwd` argument that points outside.

Then check the command's *arguments* for paths, not just the working directory — this is the part
that matters, because `cat /etc/passwd` never touches `cwd`. Keep it simple: reject any argument
that resolves outside the root. It will be imperfect (a path built inside the program won't be
caught), which is fine, because step 11 is the real answer.

While you're in the file, trim `DEFAULT_ALLOWED_COMMANDS` (`terminal_service.py:25-43`). `uv` lets
the model run `uv run python -c ...`, which is arbitrary code, and `ping` reaches the network —
both contradict the comment right above the list. Drop them.

**Verify.** Tests for the three escapes from the gap analysis: `cat /etc/passwd` with no `cwd`,
`cwd="/"`, and `cat ../../etc/hostname` from inside the root. All three should be refused.

---

## Stage 2 — Be able to measure

Before adding capability, make it possible to see what capability you added. Each of these is small
and stands alone.

### 3. Report tokens, cost and time on every run (S)

The numbers already exist — `loop.py:189` accumulates usage per turn and `loop.py:333-341` writes
it to the run record. They just never reach the user. Add `input_tokens`, `output_tokens`,
`cached_tokens` and a duration to the `RUN_FINISHED` payload (`loop.py:346-352`), and print them in
`main.py`'s `print_events`, in the `RUN_FINISHED` branch at `main.py:165-170`. Add a cost figure from
a small per-model price table next to `GROQ_MODELS` (`groq_model.py:45`).

While here, deal with the duplicate: `TokenCounter.record_usage` and `get_total`
(`context.py:43-50`) have no callers anywhere. Either wire them or delete them, so there's one way
to count tokens rather than two. I'd delete, since `loop.py` already does it.

**Verify.** Run anything and read the tokens off the last line. Check the printed total matches
what the provider reports.

### 4. Retry a turn when the model call fails transiently (S)

There's no retry anywhere in `models/`, so one rate limit ends a run. Wrap the `provider.stream()`
call in `loop.py:242` with a few attempts and a growing wait — retry on rate limits, timeouts and
dropped connections; don't retry on a bad model name or a bad key, which will never succeed. Emit
an event on each retry so it's visible rather than a mysterious pause.

This is before the evaluation runner on purpose: a sixty-task benchmark that dies at task forty on
a rate limit has to start over.

**Verify.** A test with a fake provider that fails twice then succeeds, asserting the run finishes
and reports two retries.

### 5. Turn Monitoring on and let it export (M)

`Monitoring` is built with `enabled=False` in both `service.py:62` and `loop.py:138`, and `main.py`
never passes one — so on the command line nothing is recorded at all. Default it on, and make
`shutdown()` actually write somewhere (it's currently a no-op with no callers,
`monitoring.py:64-67`). Simplest useful version: write the spans as one JSON file per run. The
OpenTelemetry exporter can come later — the `tracing` extra is already in `pyproject.toml` waiting
for it.

Also call `shutdown()` from `main.py`'s cleanup block, next to the existing `events.close()`.

**Verify.** Run once, open the JSON, confirm there's a span per turn and per tool with sensible
durations.

---

## Stage 3 — Remember things

### 6. Postgres store behind the existing interface (M)

`Database` (`database.py:20-56`) is already the right seam — eleven methods, and `MemoryDatabase`
proves the shape works. Add `postgres.py` next to it with a `PostgresDatabase` implementing the
same eleven, using `asyncpg` (already in the `postgres` extra).

Six tables, matching the code that exists rather than the old diagram: `sessions`, `messages`,
`events`, `runs`, `permission_grants`, and `memories` (step 7 uses that last one). Two things need
care. `next_sequence` must stay correct with two writers, so use a Postgres sequence or an
`UPDATE ... RETURNING` rather than read-then-write. And `load_events` must keep returning events in
sequence order, since that's the replay promise.

Ship a `schema.sql` and pick the store with an environment variable or a `--database` flag, keeping
memory as the default so nothing existing changes.

**Verify.** Run the same test suite against both stores — the interface is small enough that one
shared set of tests can cover both, which is the real payoff of this seam. Then: run the agent,
kill it, restart, and load the old conversation back.

### 7. Memory that survives between runs (M)

Two halves, both small once step 6 is in.

**Project instructions.** On `create_session`, look for a markdown file in the working folder
(`AGENT.md`) and append it to the system prompt in `PromptBuilder.build` (`config.py:52`). That's
how "always run the tests with uv" survives a restart. Nothing more to it.

**Remembered facts.** A `memories` table — id, session, kind (preference, fact, correction), text,
created, last used — plus two tools: one to write a memory, one to look them up by keyword. Inject
the handful most recently used into the first prompt. Keep it keyword-matched; no embeddings.

The reason this is worth having beyond convenience: it's the difference between an agent you
re-instruct every morning and one you can teach.

**Verify.** Tell it something in one run, start a fresh run, confirm it knows. Check `AGENT.md`
instructions actually change behaviour.

---

## Stage 4 — Reach

### 8. Subagents (M)

`Subagent` already exists as a description (`config.py:21-32`), and `AgentRuntime` already holds
several agent configs with `config_for()` to look them up (`service.py:85`). What's missing is a
tool that runs one.

Add `tools/subagent.py`: a tool whose arguments are the helper's name and the job in words. It
builds a fresh `Conversation`, a fresh `RunContext` with a low turn cap, runs `AgentLoop.run`, and
returns only the helper's final text as the tool result. The helper gets a narrower tool list via
the `allowed_tools` field that's already on `AgentConfig`.

Three things to get right. Give the helper the same cancel object so stopping the parent stops the
child. Give it a hard turn cap so it can't loop forever. And let its events through to the same bus
tagged with the parent's run, so you can see what it did — the numbered event stream makes this
easy and it's worth not skipping.

The point of this isn't speed, it's keeping the main conversation clean: a search that reads thirty
files should cost the main conversation one paragraph, not thirty files.

**Verify.** A run where the helper reads several files and the parent's conversation grows by one
tool result. Check the parent's token count doesn't include the helper's reading.

### 9. A plan the model writes down (S)

One tool that stores a list of steps, each with a status, on the session — and renders the current
list into the conversation. Nothing branches on it. There is no executor. That's the whole design,
and the restraint is the point: v0.3 failed because the plan drove execution, and the fix isn't to
avoid planning but to make the plan something the model reads rather than something that reads the
model.

**Verify.** Give it a five-part task and watch whether it still remembers part four at turn twelve.
Compare against a run with the tool disabled — this is exactly the kind of thing step 12 measures.

### 10. Run a turn's tools in parallel, with timeouts (S)

`run_tools` (`loop.py:276-329`) loops over the calls one at a time. Gather them instead. Two
constraints: keep the results in the order the model asked for them, since the conversation pairs
each result with its call; and give each tool a timeout, because concurrency without one means a
single hung call holds the whole turn. There are no timeouts anywhere in the code today, so this
adds the first.

Permission prompts need care — three tools all asking at once would be a mess. Simplest fix: ask
for all of them first, in order, then run whatever was approved in parallel.

**Verify.** A turn with three slow reads should take about as long as the slowest, not the sum.

---

## Stage 5 — The sandbox

### 11. Run tools in a container (L)

`ExecutionMode.SANDBOX` exists (`tools/base.py:28`) and `packages/sandbox` is a 2-LOC stub. This is
the structural answer to steps 1 and 2: instead of checking paths, give the tools nowhere else to
reach.

Build a small container runner in `packages/sandbox`: start a long-lived container per session with
the working folder mounted and nothing else, no network by default, with memory and CPU caps. Then
have `ToolManager.execute` (`manager.py:35`) route any tool marked `SANDBOX` through it rather than
calling `tool.execute` directly — that one branch is the whole integration, which is the payoff of
having a single gate.

Then flip the shell tool to `SANDBOX` and relax the allowlist, because the container is doing the
work the allowlist was doing badly. Keep the approval prompt regardless.

Two honest costs. It needs Docker present, so keep a direct mode for machines without it and say
which mode a run used. And it changes what "the file tools" mean — either they move into the
container too, or you accept that files and shell see the workspace differently. Decide that
explicitly rather than discovering it.

**Verify.** From inside, confirm `/etc/passwd` isn't reachable, the network is refused, and a
runaway process gets killed by the cap. Re-run the step 2 escape tests and watch them fail
structurally rather than by validation.

---

## Stage 6 — Measurement and polish

### 12. The evaluation runner and task set (L)

This is what turns the work into a result, and it's the thesis's binding constraint — but it's last
in the table because it's the biggest single piece. **Start it any time after step 5**; if two
people are working, this should run in parallel with stage 3 onward.

In `packages/evaluation`: a task file (goal, folder to run in, how to tell if it passed), a runner
that drives `AgentService` over the set, and a scorer. Fifteen or twenty tasks is plenty to start —
a few single-file reads, a few multi-file edits, a couple that need shell, a couple that should be
refused. Write results to a JSON file first; point it at the Postgres tables once step 6 lands.

Record per task: passed or failed, turns, tokens in and out, wall-clock, tools called, and whether
a permission was asked. That gives you both halves of the thesis — did it work, and what did it
cost.

Then the comparison view (C4): given two result files, show where they diverged. The event stream
is already numbered and complete, so this reads existing data rather than adding instrumentation.
This is the artifact that makes the comparison legible to a reader, and it's what lets the other
contributor's LangGraph agent be measured on the same footing — as long as both tracks use the same
tool layer and the same task set, which is the one thing that would invalidate the comparison if it
drifted.

**Verify.** Run the suite twice with the same seed and temperature 0 and confirm the pass set is
stable. An unstable suite measures noise.

### 13. Six small wins (M total, any order)

None of these need each other.

**Richer first prompt (A4).** Add a shallow folder listing and the git branch to
`PromptBuilder.build` (`config.py:52`). Saves the first few turns being spent working out where it
is. Cheapest win here.

**Prompt cache marker (A7).** The read-back is already plumbed — `Usage.cached_tokens` exists
(`conversation.py:123`), the loop reads it (`loop.py:258`) and totals it (`loop.py:412`), and it's
always zero because nothing marks the request. Mark the stable prefix (system prompt plus tool
schemas) cacheable. Do this *after* step 3 so you can see what it saved.

**Stop mid-turn (A9).** Cancel is only checked between turns (`loop.py:161`). Check it inside the
stream loop (`loop.py:242`) too. The useful version is steering — letting a correction land on the
next action rather than after the current turn finishes.

**Model-written compaction summary (A3).** `_summarize` (`context.py:149-170`) is a template that
keeps tool names and drops reasoning, so after compaction the agent can retry something it already
ruled out. Add a model-written path; keep the template as the fallback, since tests rely on it
being predictable. Worth doing *after* step 8 — with subagents the window fills more slowly and
this matters less.

**Scoped approvals (B5).** `PermissionRule.argument_pattern` already exists
(`permissions.py:73`), but grants key on tool name and session only — see `PermissionGrant.covers`
(`permissions.py:185-190`) — so "yes, writes under `./notes/` are fine" can't be said. Add an
argument pattern to the grant.

**Mark the old diagram historical.** One line at the top of
`docs/architecture/database-design.mermaid` saying it describes v0.3 and that step 6's schema
supersedes it. Costs nothing and prevents someone building twenty tables.

---

## Order, if you're doing this alone

Steps 1 and 2 in one sitting. Then 3, 4 and 5 — all small, all make the rest visible. Then 6 and 7
together, since memory is the reason to want Postgres. Then 8, 9 and 10, which all touch the loop,
so do them in sequence rather than in parallel. Then 11, which is the biggest and safest to do last
because everything before it still works without it. Start 12 as early as you can steal time for
it; every step after it lands gets a measured before-and-after instead of an opinion. 13 whenever
you want a quick win.

The one ordering I'd defend hardest: **step 3 before step 8.** Subagents are supposed to reduce
context and cost. Without token reporting you'll have no idea whether they did.

---

Every file path, method name and line number above was checked against the working tree at v0.5.0 on
2026-08-24. Line numbers will drift as soon as step 1 lands — the names are the durable part.

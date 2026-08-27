# Database normalization audit

Audit of `docs/architecture/database-design.mermaid` against 1NF – 5NF, plus the
relational-correctness problems found alongside them. Every finding below is
either fixed in the corrected diagram and in `infra/docker/postgres/schema.sql`,
or listed under [Deliberate exceptions](#deliberate-exceptions) with the reason
it stays. Boundaries the schema deliberately omits — source versioning, per-row
tenancy/RLS, and global tool infrastructure — are recorded under [Scope and
tenancy boundaries](#scope-and-tenancy-boundaries).

**Verdict on the original design:** the *shape* was right — the entity
decomposition, the task → run → step → tool-call spine, and the decision to keep
the LangGraph checkpoint tables separate from the application tables are all
sound. But it was **not in 3NF**. Nine tables carried attributes that are
functionally determined by something other than their own key, and two of them
also broke 2NF. The dominant failure mode was *copying a parent's attribute onto
a child* (`track`, `risk_level`, run metrics), which is exactly the class of
redundancy that lets two rows disagree about the same fact.

4NF and 5NF were already satisfied and remain so — see
[4NF / 5NF](#4nf--5nf-no-violations-found).

---

## Summary of findings

| # | Table | Attribute(s) | Breaks | Fix |
|---|-------|-------------|--------|-----|
| 1 | `AGENT_RUNS` | `track` | 3NF / BCNF | Dropped — lives on `agent_tasks` |
| 2 | `AGENT_RUNS` | `steps_taken`, `llm_calls`, `tool_calls`, `total_tokens`, `cost` | Derived redundancy | Replaced by `v_run_metrics`; new `llm_calls` table gives tokens/cost a source of truth |
| 3 | `AGENT_RUNS` | `duration_ms` | Derived redundancy | `GENERATED ALWAYS AS` column |
| 4 | `AGENT_RUNS` | `langfuse_trace_id` | Redundancy vs `TRACE_REFS` | Dropped — `trace_refs` is the single home |
| 5 | `PLAN_STEPS` | `verified` | Derived redundancy | Dropped — derived from `verification_results` |
| 6 | `PLAN_STEPS` | *(missing)* `summary`, `reasoning`, `requires_remediation` | Missing entity | New `plans` table (a run replans per phase) |
| 7 | `TOOL_CALLS` | `risk_level` | 3NF / BCNF | Kept, but re-anchored on `risk_policy_id` so it records a *decision*, not a derivation |
| 8 | `TOOL_CALLS`, `PLAN_STEPS` | `tool_name` free text | Redundancy, no integrity | New `tools` + `mcp_servers`; FK `tool_id` |
| 9 | `TOOL_CALLS` | `duration_ms` | Derived redundancy | `GENERATED ALWAYS AS` column |
| 10 | `APPROVAL_REQUESTS` | `risk_level`, `requested_action` | 3NF / BCNF | Dropped — determined by `tool_call_id` / `plan_step_id` |
| 11 | `APPROVAL_REQUESTS` | `resolved_by` free text | No integrity | FK `resolved_by_actor_id` → new `actors` |
| 12 | `SANDBOX_SESSIONS` | `memory_limit_mb`, `cpu_limit`, `time_limit_s` | 3NF / BCNF | Dropped — determined by the run's config snapshot |
| 13 | `RUN_CONFIG_SNAPSHOTS` | all 8 config columns | Redundancy + wrong cardinality | Content-addressed `config_snapshots`, shared by N runs |
| 14 | `TRACE_REFS` | `session_id` | 3NF | Moved to `agent_threads.trace_session_id` |
| 15 | `EVALUATION_RESULTS` | `track` | **2NF** | Dropped — determined by `evaluation_run_id` alone |
| 16 | `EVALUATION_RESULTS` | `steps`, `latency_ms`, `tokens`, `tool_calls`, `cost` | 3NF / BCNF | Dropped — determined by `agent_run_id` |
| 17 | `EVALUATION_RESULTS` | `metrics jsonb` | **1NF** | New `evaluation_scores(result_id, metric, value)` |
| 18 | `EVALUATION_SUITES` | `(name, version)` | Missing candidate key | `UNIQUE (name, version)` |
| 19 | `MEMORY_POINTS`, `KNOWLEDGE_CHUNKS` | `content` + `payload` | 1NF + dual system of record | Postgres `memory_items` / `knowledge_chunks` are authoritative; Qdrant mirrors by id |
| 20 | *everywhere* | no `(run_id, sequence)` / `(plan_id, step_number)` uniqueness | Missing candidate keys | Declared as `UNIQUE` |

---

## 1NF — atomic attributes

**Violation found: `EVALUATION_RESULTS.metrics jsonb`.** This is a repeating group
of `(metric_name, value)` pairs. Unlike an event payload, eval scores are a
*known, closed, queryable* set of facts — the whole point of the table is to
compare them across tracks (`WHERE metric = 'latency_ms' GROUP BY track`).
Burying them in JSONB means no type, no unit, no per-metric index, and no way to
constrain a metric to appear once. Extracted to:

```
evaluation_scores(id, result_id FK, metric, value numeric, comment)
UNIQUE (result_id, metric)
```

**Violation found: `MEMORY_POINTS` / `KNOWLEDGE_CHUNKS` store `content` *and*
`payload`.** In Qdrant the chunk text normally lives *inside* the payload, so the
design stores the same string twice with nothing to keep the copies equal. The
corrected design makes Postgres authoritative for the text and metadata and keeps
Qdrant to `(point_id, vector)` plus the small filterable subset.

**Not a repeating group: `config_snapshots.*_config`.** The eight `*_config` JSONB
columns look like one at a glance, but they are *distinct, named* attributes
(`llm` / `execution` / `sandbox` / `permissions` / …), not the same attribute
repeated, so 1NF is not the objection. They are intentionally non-relational:
immutable config documents, stored and read whole, with nothing queried
relationally living inside them. The real problem was never their shape but their
*cardinality* — the entire `AgentConfig` was re-stored per run — which #13 fixes by
content-addressing.

## 2NF — no partial dependency on part of a candidate key

Most tables use a single-column surrogate `uuid` primary key, so a 2NF violation
cannot arise *on the primary key*. It can still arise on a **candidate key**, and
one does:

**`EVALUATION_RESULTS`** has the candidate key `(evaluation_run_id, case_id)` —
one result per case per eval run. Its `track` column depends on
`evaluation_run_id` **alone**, i.e. on part of that key. Textbook partial
dependency. Dropped; `track` is read from `evaluation_runs`.

`PLAN_STEPS (run_id, step_number)`, `AGENT_EVENTS (run_id, sequence_number)` and
the LangGraph composite keys were checked and are clean — every non-key attribute
depends on the whole key.

## 3NF / BCNF — no transitive dependency, every determinant a superkey

This is where the original design failed, in seven places. The pattern is
identical each time: a child copies an attribute of its parent, so the determinant
is a *foreign key* rather than a superkey of the table it sits in.

**1. `AGENT_RUNS.track`.** `run_id → task_id → track`. A run cannot be on a
different track from its task, but the schema permits it. Dropped.

**2. `APPROVAL_REQUESTS.risk_level` and `.requested_action`.**
`tool_call_id → risk_level`, and `requested_action` is just the tool name plus
arguments already stored on the step and the call. Both dropped; the approval row
now carries only what is genuinely its own — status, reason, who resolved it, when.

**3. `TRACE_REFS.session_id`.** The Langfuse session id *is* the thread id — see
[`langgraph_agent.py:120`](../../packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L120),
which sets `langfuse_session_id` from `task.thread_id`. So
`run_id → task_id → thread_id → session_id`: a two-hop transitive dependency, and
every run of the same thread re-stores the same value. Moved to
`agent_threads.trace_session_id`.

**4. `EVALUATION_RESULTS.{steps, latency_ms, tokens, tool_calls, cost}`.** All
determined by `agent_run_id`. This is the most dangerous instance in the design:
the same five numbers exist in `agent_runs` *and* here, and a benchmark report
built from the wrong copy is silently wrong. Dropped; joined from the run.

**5. `SANDBOX_SESSIONS.{memory_limit_mb, cpu_limit, time_limit_s}`.** These come
from `SandboxConfig` ([`config.py:51`](../../packages/common/src/common/config.py#L51)),
which is already captured in the run's config snapshot. `run_id` is not a superkey
of `sandbox_sessions` (a run may create several), so this is a BCNF violation as
well as a duplicate of the snapshot. Dropped.

**6. `AGENT_RUNS.langfuse_trace_id` vs the whole `TRACE_REFS` table.** The same
fact modelled twice, once as a column and once as a table. Kept the table (it
generalises to other providers and to multiple traces per run); dropped the column.

**7. `TOOL_CALLS.risk_level` — kept, but re-anchored.** Risk is a *pure function*
of `(tool_name, arguments)`:
[`RiskClassifier.classify`](../../packages/common/src/common/risk.py#L66) is
deterministic with no side effects. So `{tool_name, arguments} → risk_level`, and
that determinant is not a superkey — a BCNF violation on the letter of the law.

It is still the right column to keep, because the honest dependency is
`{tool_name, arguments, ruleset} → risk_level`, and `DEFAULT_RULES` changes over
time. Reclassifying an old call under today's rules would rewrite history and
break the audit trail. The fix is therefore to make the *third* determinant
explicit rather than to delete the column: `tool_calls.risk_policy_id` FKs to a
`risk_policies` table holding the versioned ruleset, and `risk_reason` stores the
matched rule from `RiskClassifier.explain`. The row now records **a decision that
was made**, which is an irreducible fact, not a derivable one.

### Derived-data redundancy (not an FD violation, but the same anomaly)

Storing a computed aggregate is not a normal-form violation in Codd's sense —
normal forms constrain dependencies *among attributes of a relation*, and an
aggregate over a child table is not one. It produces the identical failure mode
though, so it is treated the same way here.

- `duration_ms` on `agent_runs` and `tool_calls` is `finished_at - started_at`.
  Now a `GENERATED ALWAYS AS ... STORED` column: same read performance, no way to
  disagree.
- `PLAN_STEPS.verified` is `EXISTS (verification_results WHERE result='verified')`.
  Dropped. Note the runtime has the same redundancy —
  [`state.py:37-45`](../../packages/agent-langgraph/src/agent_langgraph/graph/state.py#L37-L45)
  carries both `verified: bool` and `verification: VerificationResult | None`,
  where `verified == (verification is VERIFIED)`. That is fine in transient graph
  state; it should not reach disk.
- `AGENT_RUNS.{steps_taken, llm_calls, tool_calls, total_tokens, cost}` — replaced
  by the `v_run_metrics` view.

  Worth calling out: **four of those five were not derivable at all** in the
  original design. `steps_taken` and `tool_calls` could be counted from their
  child tables, but nothing recorded individual model invocations, so `llm_calls`,
  `total_tokens` and `cost` had no source of truth anywhere in the schema — which
  is why
  [`langgraph_agent.py:232-235`](../../packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L232-L235)
  hardcodes them to zero and defers to Langfuse. The corrected design adds an
  `llm_calls` table (one row per model call: provider, model, prompt/completion
  tokens, cost, latency, trace observation id). Cost then aggregates from real
  rows, and per-node/per-model cost attribution becomes a query instead of a
  dashboard visit.

### `RUN_CONFIG_SNAPSHOTS` → content-addressed `config_snapshots`

Two problems. The cardinality was drawn `AGENT_RUNS ||--o{ RUN_CONFIG_SNAPSHOTS`,
i.e. one run to *many* snapshots, when a run executes under exactly one config.
And every run stored the full `AgentConfig` even though a thousand consecutive
runs share it byte for byte.

Fixed by content-addressing: `config_snapshots(id, content_hash UK, …)` with
`agent_runs.config_snapshot_id` and `evaluation_runs.config_snapshot_id` both
pointing at it. Insert is an upsert on `content_hash`. `content_hash` is a genuine
candidate key that determines every other column, so the table is BCNF-clean, N
runs share one row, and "which runs used this exact config?" becomes an indexed
equality lookup instead of a JSONB comparison.

## 4NF / 5NF — no violations found

Checked and clean; nothing was changed for these.

- **4NF.** A 4NF violation needs two independent multi-valued facts in one
  relation. The candidates — a task's attachments, a run's events, a run's tool
  calls, a suite's cases — are each already in their own table with a single
  multi-valued dependency. `task_attachments` being extracted rather than sitting
  as an array on `agent_tasks` is precisely what keeps this clean.
- **5NF.** No relation here is a lossless join of smaller projections. The only
  three-way relationship, `evaluation_results(evaluation_run_id, case_id,
  agent_run_id)`, is not decomposable: the eval run and the case jointly identify
  which agent run was launched, so the ternary fact carries information that no
  pair of binary projections reconstructs. (The `suite_id` also on that table is
  not a fourth fact but a carried consistency key — see
  [Deliberate exceptions](#deliberate-exceptions).)

---

## Correctness problems found alongside the normalization work

These are not normal-form issues, but they would corrupt or block writes, so they
are fixed in the same pass.

**1. `AGENT_TASKS.status` listed the wrong enum.** The diagram annotated it
`created | pending | running | completed | failed | interrupted` — that is
`RunStatus`. The code's task-level enum is `TaskStatus` = `planning | executing |
verifying | responding | completed | failed | skipped | interrupted`
([`enums.py:28`](../../packages/common/src/common/enums.py#L28)). Two different
enums were being described by one column. Corrected: `agent_tasks.status` is
`TaskStatus`, `agent_runs.status` is `RunStatus`, and both are real Postgres enum
types so a mismatch fails at insert.

**2. `APPROVAL_REQUESTS.tool_call_id` was unfillable at insert time.** The human
gate fires *before* the tool runs —
[`executor.py:113`](../../packages/agent-langgraph/src/agent_langgraph/nodes/executor.py#L113)
calls `interrupt()` and only invokes the tool if the decision comes back
approved. So at the moment an approval row is created there is no tool call to
point at, and if the human denies, there never will be one. Approval is therefore
re-anchored on `plan_step_id` (which does exist), with `tool_call_id` nullable and
backfilled after an approved call executes. A partial unique index allows only one
`pending` approval per step.

**3. `uuid` ↔ `text` mismatch on the checkpoint join.** `AGENT_THREADS.id` was
`uuid` while `LG_CHECKPOINTS.thread_id` is `TEXT` — the drawn relationship could
not be a real foreign key, and even the join needs an explicit cast. Worse, the
target tables are created and migrated by `PostgresSaver`, which will never honour
an application FK. `agent_threads.id` is now `text` holding a canonical UUID
string, so it is the same value LangGraph is given as `configurable.thread_id`
([`langgraph_agent.py:128`](../../packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L128)),
and the relationship is documented as a logical join with no enforced FK.

**4. The `LG_*` tables did not match the library.** Verified against the installed
`langgraph.checkpoint.postgres.base.MIGRATIONS`:

- `checkpoints` has **no `created_at`** column; the diagram invented one.
- `checkpoint_writes` was **missing `task_path TEXT NOT NULL DEFAULT ''`** (added
  by the tenth migration).
- `checkpoint_migrations(v INTEGER PK)` was missing entirely.
- `checkpoint_blobs.blob` is **nullable** (the fifth migration drops the NOT NULL).
- The relationship `LG_CHECKPOINT_BLOBS ||--o{ LG_CHECKPOINT_WRITES` **does not
  exist**. There are no foreign keys at all between these three tables. The real
  linkage is `checkpoints → checkpoint_blobs`, joined on
  `(thread_id, checkpoint_ns, channel, version)` where channel/version come from
  `checkpoint -> 'channel_versions'`. Corrected, and the whole group is marked
  library-owned: *do not reshape, do not add FKs, do not hand-migrate*.

**5. Nowhere to persist findings or the workflow phase.** `findings` are the one
piece of state the code explicitly designs for durability — they are appended
never replaced, precisely so they survive the replan that swaps out `plan`
([`state.py:130-133`](../../packages/agent-langgraph/src/agent_langgraph/graph/state.py#L130-L133),
[`phase_transition.py:21-47`](../../packages/agent-langgraph/src/agent_langgraph/nodes/phase_transition.py#L21-L47)) —
yet the schema had no table for them, and no column for `workflow_phase`. Added
`run_phases` and `run_findings`.

**6. No `plans` table, so a replan overwrote its predecessor.** A run produces
*several* plans (one per phase, plus reflection replans), and `AgentPlan` carries
`summary`, `reasoning` and `requires_remediation`
([`state.py:51-75`](../../packages/agent-langgraph/src/agent_langgraph/graph/state.py#L51-L75))
which had nowhere to live. With steps hanging directly off the run, the second
plan's `step_number = 1` collides with the first's and the two plans become
indistinguishable. Added `plans(run_id, revision, phase_id, summary, reasoning,
requires_remediation)`; steps hang off a plan.

**7. No `actors` table.** `resolved_by` was free text, and a thread had no owner —
so there was no way to record who approved a risky action beyond a name string,
and no tenancy boundary. Added `actors`, referenced by
`agent_threads.owner_actor_id` and `approval_requests.resolved_by_actor_id`.
`langgraph_agent.py` already threads a `user_id` through to Langfuse
([`langgraph_agent.py:123`](../../packages/agent-langgraph/src/agent_langgraph/orchestrator/langgraph_agent.py#L123));
now it has somewhere to land.

**8. Thread-scoped memory defeated cross-thread recall.** `MEMORY_POINTS.thread_id`
was a plain FK to `agent_threads`, which makes every memory thread-local. A
`preference` memory that cannot outlive the conversation that produced it is not
semantic memory. The scope actually depends on `memory_type` — preferences are
actor-scoped, summaries are thread-scoped — so the corrected design carries an
explicit `scope` discriminator with a CHECK enforcing the full disjoint binding:
actor-scoped rows have an actor and no thread, thread-scoped rows have a thread and
no actor, and global rows have neither.

**9. Two systems of record for vectors.** Drawing Qdrant collections as tables
with FKs into Postgres implies integrity that no database enforces; a failed
second write leaves an orphan with nothing to detect it. Postgres is now
authoritative for memory/knowledge text and metadata, Qdrant holds
`point_id = <postgres id>` plus the vector, and the Qdrant side is drawn as
clearly-labelled mirror entities with no FK notation.

---

## Deliberate exceptions

Kept knowingly. Each is a case where full normalization would cost more than the
anomaly it prevents.

| Kept | Why |
|------|-----|
| `metadata jsonb` on threads, tasks, tools, sources | Genuinely open key/value set supplied by callers. Rule: **nothing queried relationally may live in here.** The moment an attribute is filtered or joined on, it gets a column. |
| `agent_events.payload jsonb` | The payload shape varies by `event_type` — an EER specialization over `AgentEvent` ([`events.py`](../../packages/common/src/common/events.py)). Modelled as single-table-with-discriminator, which is the standard relational mapping when subtypes are numerous, thin, and read as a stream. |
| `plan_steps.arguments`, `tool_calls.arguments` | Per-tool JSON Schema; cannot be columns. Both are kept because they are *different facts* — planned arguments vs. the arguments actually sent on a given attempt (retries can differ). |
| `config_snapshots.*_config jsonb` | An immutable audit blob, never queried field-by-field. Content-addressed, so it is stored once regardless of how many runs share it. |
| `plan_steps.run_id`, `tool_calls.run_id` | Redundant against the `plan → run` path, but **constraint-enforced**: composite FKs `(run_id, plan_id) → plans(run_id, id)` and `(run_id, plan_step_id) → plan_steps(run_id, id)` make disagreement impossible. This buys a single-predicate partition key for the hottest query in the system ("everything for this run") without the anomaly a bare copy would create. Required anyway, since a ReAct-style tool call has no plan step. |
| `evaluation_results.suite_id` | Redundant against both `evaluation_run_id → suite` and `case_id → suite`, but **constraint-enforced**: composite FKs `(evaluation_run_id, suite_id) → evaluation_runs(id, suite_id)` and `(case_id, suite_id) → evaluation_cases(id, suite_id)` force the run and the case into the *same* suite. Without the carried column a bare pair of single-column FKs would let a suite-A run be scored against a suite-B case. The two parents therefore also carry a `UNIQUE (id, suite_id)` so the composite FKs have a target. |
| `v_run_metrics` as a view | Correct by construction. If run-list latency ever needs it, promote to a materialized view or a rollup table with a trigger — but only then, and the numbers stay derived either way. |
| The `LG_*` group | Library-owned and denormalized on purpose (`thread_id` on every row, serialized blobs). Reshaping it breaks `PostgresSaver`. |

---

## Scope and tenancy boundaries

Three things the schema deliberately does *not* model. Each was raised in review;
each is a boundary, not an oversight, so it is recorded here rather than "fixed".

**Knowledge sources are re-indexed in place, not versioned.** `knowledge_sources`
carries `source_uri` (`UNIQUE`, its identity), a mutable `content_hash`, and a
single `indexed_at`; there is no `version` column or history table, and
`knowledge_chunks` cascade-delete with the source. Re-indexing a changed source
therefore overwrites — `content_hash`/`indexed_at` advance and the old chunks are
replaced. Intentional: the knowledge base is a *current-state* retrieval index,
not an archive, and nothing in the system reads "what did this document say last
month". `content_hash` exists to *detect* a change cheaply and rebuild (see its
comment in `schema.sql`), not to retain the prior revision. Point-in-time
knowledge, if ever required, is additive — a `knowledge_source_versions` table, or
a `superseded_by_id` mirroring the one already on `memory_items` — not a reshape of
what exists.

**Tenancy is derived through the ownership chain, not stamped on every row.**
`agent_threads.owner_actor_id` is the single tenancy anchor; the run spine reaches
its owner by following keys —
`tool_calls → agent_runs → agent_tasks → agent_threads → owner_actor_id` — and
carries no `actor_id`/`tenant_id` of its own. That is the normalized choice:
ownership is a fact about the thread, and copying it onto every task, run, step and
call is exactly the parent-attribute-on-child redundancy this audit removes
elsewhere (two rows could then disagree about who owns the same run). The cost — an
ownership predicate is a join, not a column — is accepted at the current scale,
with two consequences recorded for when it changes:

- **No row-level security here.** There are no `CREATE POLICY` statements. When the
  deployment goes multi-tenant, RLS belongs on the tables that *hold* the anchor
  (`agent_threads`, and `memory_items.owner_actor_id` for actor-scoped memory),
  with descendants mediated by the join. If join-per-check proves too costly under
  RLS, the fix is a constraint-enforced `owner_actor_id` copy on the hot tables —
  the same composite-FK pattern already used for `plan_steps.run_id` and
  `evaluation_results.suite_id` — never a bare copy. It is omitted now because it
  enforces nothing until a policy exists.
- **Knowledge is intentionally un-owned.** `knowledge_sources` has no
  `owner_actor_id`: the knowledge base is shared infrastructure, not actor-scoped
  data. Per-tenant knowledge, if required, is a new column and access rule, not a
  migration of existing ownership.

**MCP servers are global infrastructure, so `mcp_servers.name` is globally
`UNIQUE`.** The `UNIQUE (name)` asserts one server per logical name across the
whole deployment, with `tools UNIQUE (server_id, name)` beneath it — the tool
catalogue is shared, not per-tenant. This matches how the rows are populated:
upserted from the deployment's own MCP configuration on discovery, never created by
end users. Should servers ever become tenant-scoped, the uniqueness key changes to
`(owner_actor_id, name)` and `tools`/`tool_calls` inherit the scope through their
existing `server_id` FK unchanged — which is the reason to anchor tenancy on the
server rather than repeat it downward.

---

## Which representation to generate the schema from

The question of *EER (Chen) vs. crow's-foot* matters here because only one of
them maps mechanically to DDL.

**Crow's-foot ER, at the logical/physical level, is the right source** — kept as
`database-design.mermaid` and implemented in `infra/docker/postgres/schema.sql`.
Every element of the notation has exactly one SQL counterpart: an entity is a
table, an attribute is a column, `PK`/`FK`/`UK` are the three key constraint
kinds, and the `||--o{` glyphs pin down both cardinality *and* optionality — which
is precisely the `NOT NULL` decision on the foreign key. The translation is
lossless in the direction you need it, and it round-trips: you can read the
diagram back off the schema.

**EER's distinctive constructs do not survive the trip.** Specialization
hierarchies, disjoint/overlapping and total/partial constraints, union types,
multi-valued and composite attributes, and aggregation are all conceptual devices
with *several* valid relational mappings each. A Chen-style EER diagram is the
better tool for arguing about the domain — but it defers the exact decisions that
generate DDL, so generating a schema from it means making those choices twice.

The workable answer is to use EER thinking and record the *outcome* in crow's-foot
form. Both EER constructs present in this domain are handled that way:

- **`AgentEvent` specialization** (`PlanningStarted`, `ToolStarted`,
  `ToolFinished`, `AgentFinished`) → single table + `event_type` discriminator +
  `payload`. Chosen over table-per-subtype because subtypes are thin, open-ended,
  and always read as one ordered stream per run.
- **Memory-type specialization** with scope varying by subtype → single table +
  `scope` discriminator + a CHECK that enforces the disjointness the EER diagram
  would have drawn as a `d` circle.

The one thing the crow's-foot diagram genuinely cannot express is *inter-column
constraints* — "resolved implies resolved_at is set", "actor-scoped implies an
actor". Those live as `CHECK` constraints in the DDL and as `%%` notes in the
diagram, and the DDL is authoritative for them.

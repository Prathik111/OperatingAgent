-- =============================================================================
-- OperatingAgent - PostgreSQL schema (system of record)
-- =============================================================================
-- Generated from docs/architecture/database-design.mermaid (crow's-foot) and
-- audited in docs/architecture/database-normalization.md. Crow's-foot is the
-- source of truth for the physical schema because every glyph maps to exactly
-- one SQL construct (entity->table, attribute->column, PK/FK/UK->constraint,
-- ||--o{ cardinality->the NOT NULL decision on the FK); see the audit's
-- "Which representation to generate the schema from" section.
--
-- Target: PostgreSQL 14+.
-- Apply: mount into the postgres container's /docker-entrypoint-initdb.d/, or
--        run once with `psql "$DATABASE_URL" -f schema.sql`. Idempotent enough
--        to re-run against an empty database; not a migration tool.
--
-- Deliberately NOT created here:
--   * The langgraph.checkpoint.postgres tables (checkpoints, checkpoint_blobs,
--     checkpoint_writes, checkpoint_migrations). PostgresSaver creates AND
--     migrates them from its own MIGRATIONS list; hand-authoring or adding FKs
--     to them breaks the library. agent_threads.id is TEXT holding a canonical
--     UUID string precisely so it equals the value handed to LangGraph as
--     configurable.thread_id and joins to checkpoints.thread_id as a *logical*
--     join with no enforced FK.
--   * The Qdrant collections (qdrant_memory_vectors, qdrant_knowledge_vectors).
--     They live in an external vector store, not Postgres. Postgres is the
--     system of record for the text/metadata; Qdrant mirrors by point_id =
--     <postgres id> and holds only the vector plus a filterable payload subset.
--
-- Convention: a value set that is a first-class enum in the domain model
-- (common/enums.py) becomes a Postgres ENUM type, so a mismatch fails at insert.
-- An ad-hoc / open-ended string set is a TEXT column with a CHECK, which is
-- cheaper to evolve than ALTER TYPE.
-- =============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid() (core since PG13; kept for portability)

-- -----------------------------------------------------------------------------
-- Enum types (mirror common/enums.py exactly)
-- -----------------------------------------------------------------------------
CREATE TYPE agent_track          AS ENUM ('native', 'langgraph');
CREATE TYPE run_status           AS ENUM ('created', 'pending', 'running', 'completed', 'failed', 'interrupted');
CREATE TYPE task_status          AS ENUM ('planning', 'executing', 'verifying', 'responding', 'completed', 'failed', 'skipped', 'interrupted');
CREATE TYPE risk_level           AS ENUM ('safe', 'review', 'blocked');
CREATE TYPE verification_verdict AS ENUM ('verified', 'not_verified', 'skipped');
CREATE TYPE workflow_phase       AS ENUM ('investigate', 'remediate', 'complete');

-- -----------------------------------------------------------------------------
-- Shared trigger: keep updated_at honest without trusting the writer
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Identity and conversation
-- =============================================================================

CREATE TABLE actors (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind        text NOT NULL,                       -- human | agent | system
    external_id text UNIQUE,                          -- idp subject, api key id, or service name
    display_name text,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT actors_kind_valid CHECK (kind IN ('human', 'agent', 'system'))
);

-- id is TEXT holding a canonical UUID string (not uuid): it is the exact value
-- handed to LangGraph as configurable.thread_id, and checkpoints.thread_id is
-- TEXT. Matching the type keeps that logical join cast-free.
CREATE TABLE agent_threads (
    id                 text PRIMARY KEY DEFAULT (gen_random_uuid())::text,
    external_thread_id text UNIQUE,
    owner_actor_id     uuid NOT NULL REFERENCES actors (id) ON DELETE RESTRICT,
    title              text,
    trace_session_id   text,                          -- Langfuse session id: one per thread, not per run
    metadata           jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now()
);

-- track lives here and only here; agent_runs no longer copies it.
CREATE TABLE agent_tasks (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id  text NOT NULL REFERENCES agent_threads (id) ON DELETE CASCADE,
    goal       text NOT NULL,
    track      agent_track NOT NULL,
    status     task_status NOT NULL DEFAULT 'planning',
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE task_attachments (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id    uuid NOT NULL REFERENCES agent_tasks (id) ON DELETE CASCADE,
    kind       text NOT NULL,                          -- file | url | image | snippet
    uri        text NOT NULL,
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT task_attachment_kind_valid CHECK (kind IN ('file', 'url', 'image', 'snippet'))
);

-- =============================================================================
-- Run execution spine
-- =============================================================================

-- Content-addressed: N runs sharing one AgentConfig store one row. content_hash
-- is a candidate key determining every config column, which is what makes the
-- table BCNF-clean. Insert is an upsert on content_hash. Shared by agent_runs
-- and evaluation_runs.
CREATE TABLE config_snapshots (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash       text NOT NULL UNIQUE,           -- sha256 over the canonical JSON of all sections
    llm_config         jsonb,
    execution_config   jsonb,
    sandbox_config     jsonb,
    permissions_config jsonb,
    checkpoint_config  jsonb,
    tracing_config     jsonb,
    behaviour_config   jsonb,
    prompts_config     jsonb,
    created_at         timestamptz NOT NULL DEFAULT now()
);

-- Counters (steps_taken / llm_calls / tool_calls / total_tokens / cost) and
-- track / langfuse_trace_id were dropped: metrics are aggregates exposed as
-- v_run_metrics, track lives on the task, and the trace lives in trace_refs.
CREATE TABLE agent_runs (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id            uuid NOT NULL REFERENCES agent_tasks (id) ON DELETE CASCADE,
    attempt            integer NOT NULL DEFAULT 1,     -- a retried task gets attempt 2
    config_snapshot_id uuid NOT NULL REFERENCES config_snapshots (id) ON DELETE RESTRICT,
    status             run_status NOT NULL DEFAULT 'created',
    output             text,                            -- responder answer
    last_error         text,
    retry_count        integer NOT NULL DEFAULT 0,
    metadata           jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at         timestamptz,
    finished_at        timestamptz,
    duration_ms        integer GENERATED ALWAYS AS
                           ((EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000)::integer) STORED,
    CONSTRAINT uq_agent_runs_task_attempt UNIQUE (task_id, attempt),
    CONSTRAINT agent_runs_attempt_positive CHECK (attempt > 0),
    CONSTRAINT agent_runs_retry_count_nonneg CHECK (retry_count >= 0)
);

-- workflow_phase now has a home, so a two-phase run (investigate then remediate)
-- is distinguishable from a single-phase one.
CREATE TABLE run_phases (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id       uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    sequence     integer NOT NULL,                     -- workflow-assigned order within a run (unique, not monotonic-enforced)
    phase        workflow_phase NOT NULL,
    entry_reason text,
    entered_at   timestamptz NOT NULL DEFAULT now(),
    exited_at    timestamptz,
    CONSTRAINT uq_run_phases_run_sequence UNIQUE (run_id, sequence),
    CONSTRAINT uq_run_phases_run_id UNIQUE (run_id, id),        -- composite-FK target (plans.phase_id, run_findings.phase_id)
    CONSTRAINT run_phases_sequence_nonneg CHECK (sequence >= 0)
);

-- A run produces several plans (one per phase, plus reflection replans). The
-- UNIQUE (run_id, id) is the target of plan_steps' composite FK below. phase_id
-- is a composite FK to run_phases(run_id, id) so a plan cannot point at a phase
-- belonging to a different run.
CREATE TABLE plans (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id               uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    phase_id             uuid NOT NULL,
    revision             integer NOT NULL,             -- workflow-assigned revision within a run
    summary              text,
    reasoning            text,
    requires_remediation boolean NOT NULL DEFAULT false,
    created_at           timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_plans_run_revision UNIQUE (run_id, revision),
    CONSTRAINT uq_plans_run_id UNIQUE (run_id, id),     -- composite-FK target
    CONSTRAINT plans_revision_nonneg CHECK (revision >= 0),
    CONSTRAINT fk_plans_phase FOREIGN KEY (run_id, phase_id)
        REFERENCES run_phases (run_id, id) ON DELETE CASCADE
);

-- =============================================================================
-- Tool access (defined before plan_steps/tool_calls, which reference it)
-- =============================================================================

CREATE TABLE mcp_servers (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name       text NOT NULL UNIQUE,                   -- gateway | file-server | terminal-server | ...
    base_url   text,
    enabled    boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Upserted on MCP discovery, never hand-maintained; a tool_id FK replaces the
-- free-text tool_name that used to be repeated across plan_steps and tool_calls.
CREATE TABLE tools (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    server_id          uuid NOT NULL REFERENCES mcp_servers (id) ON DELETE CASCADE,
    name               text NOT NULL,
    description        text,
    input_schema       jsonb,
    output_schema      jsonb,
    default_risk_level risk_level,                     -- advertised level, not a verdict
    discovered_at      timestamptz NOT NULL DEFAULT now(),
    last_seen_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_tools_server_name UNIQUE (server_id, name)
);

-- Risk is a pure function of (tool, arguments, ruleset); naming the ruleset
-- version is what turns tool_calls.risk_level from a redundant derivation into a
-- record of a decision actually made.
CREATE TABLE risk_policies (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    version    text NOT NULL UNIQUE,
    ruleset    jsonb NOT NULL,                         -- ordered RiskRule list, most dangerous first
    created_at timestamptz NOT NULL DEFAULT now()
);

-- run_id is redundant against plan_id -> plans.run_id but the composite FK
-- (run_id, plan_id) -> plans(run_id, id) makes the two unable to disagree; it
-- earns its place as the partition key for every per-run query. verified was
-- dropped (derivable from verification_results); tool_name became tool_id.
CREATE TABLE plan_steps (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id     uuid NOT NULL,
    run_id      uuid NOT NULL,
    step_number integer NOT NULL,
    description text,
    tool_id     uuid REFERENCES tools (id) ON DELETE RESTRICT,  -- null for a reasoning-only step
    arguments   jsonb,                                 -- planned args; tool_calls holds what was sent
    status      run_status NOT NULL DEFAULT 'created', -- matches PlanStep.status in graph state
    output      text,
    started_at  timestamptz,
    finished_at timestamptz,
    CONSTRAINT uq_plan_steps_plan_step_number UNIQUE (plan_id, step_number),
    CONSTRAINT uq_plan_steps_run_id UNIQUE (run_id, id),        -- composite-FK target
    CONSTRAINT plan_steps_step_number_nonneg CHECK (step_number >= 0),
    CONSTRAINT fk_plan_steps_plan FOREIGN KEY (run_id, plan_id)
        REFERENCES plans (run_id, id) ON DELETE CASCADE
);

CREATE TABLE sandbox_sessions (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    container_id   text UNIQUE,                         -- null until the container exists
    workspace_path text,
    status         text NOT NULL DEFAULT 'creating',    -- creating | ready | destroyed | failed
    created_at     timestamptz NOT NULL DEFAULT now(),
    destroyed_at   timestamptz,
    CONSTRAINT sandbox_status_valid CHECK (status IN ('creating', 'ready', 'destroyed', 'failed')),
    CONSTRAINT uq_sandbox_sessions_run_id UNIQUE (run_id, id)    -- composite-FK target (tool_calls.sandbox_session_id)
);

-- (run_id, plan_step_id) is a composite FK to plan_steps(run_id, id), so the
-- redundant run_id cannot disagree with the step's run. plan_step_id is nullable
-- (a ReAct-style call has no plan step); the composite FK is MATCH SIMPLE, so it
-- simply does not fire when plan_step_id is NULL -- which is exactly why the
-- direct run_id FK is *also* needed, to validate a ReAct call's run.
-- sandbox_session_id follows the same pattern: (run_id, sandbox_session_id) is a
-- composite FK to sandbox_sessions(run_id, id), pinning the session to this run.
CREATE TABLE tool_calls (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id             uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    plan_step_id       uuid,
    tool_id            uuid NOT NULL REFERENCES tools (id) ON DELETE RESTRICT,
    sandbox_session_id uuid,
    attempt            integer NOT NULL DEFAULT 1,      -- retry number within the step
    arguments          jsonb,                           -- what was actually sent
    risk_policy_id     uuid REFERENCES risk_policies (id) ON DELETE RESTRICT,
    risk_level         risk_level,
    risk_reason        text,                            -- the rule that matched
    success            boolean,
    output             jsonb,
    error              text,
    trace_span_id      text,
    started_at         timestamptz,
    finished_at        timestamptz,
    duration_ms        integer GENERATED ALWAYS AS
                           ((EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000)::integer) STORED,
    CONSTRAINT uq_tool_calls_run_id UNIQUE (run_id, id),        -- composite-FK target (approval_requests.tool_call_id)
    CONSTRAINT tool_calls_attempt_positive CHECK (attempt > 0),
    CONSTRAINT fk_tool_calls_step FOREIGN KEY (run_id, plan_step_id)
        REFERENCES plan_steps (run_id, id) ON DELETE CASCADE,
    CONSTRAINT fk_tool_calls_sandbox FOREIGN KEY (run_id, sandbox_session_id)
        REFERENCES sandbox_sessions (run_id, id) ON DELETE CASCADE
);

-- Gives llm_calls / total_tokens / cost a source of truth: without it the
-- orchestrator hardcodes them to zero. total_tokens and duration_ms are derived.
CREATE TABLE llm_calls (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id               uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    plan_step_id         uuid,                          -- null for a node-level call (planner/responder); run-scoped via the composite FK below
    node_name            text,                          -- planner | verifier | responder | error_handler
    provider             text,
    model                text,
    prompt_tokens        integer,
    completion_tokens    integer,
    total_tokens         integer GENERATED ALWAYS AS (prompt_tokens + completion_tokens) STORED,
    cost                 numeric,
    trace_observation_id text,
    error                text,
    started_at           timestamptz,
    finished_at          timestamptz,
    duration_ms          integer GENERATED ALWAYS AS
                             ((EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000)::integer) STORED,
    CONSTRAINT fk_llm_calls_step FOREIGN KEY (run_id, plan_step_id)
        REFERENCES plan_steps (run_id, id) ON DELETE CASCADE
);

-- Findings are the one piece of graph state designed for durability: appended
-- never replaced, so they survive the replan that swaps out the plan.
CREATE TABLE run_findings (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    phase_id       uuid,
    plan_step_id   uuid,
    description    text,                                -- what was being investigated
    detail         text,                                -- the step output that matters
    source_tool_id uuid REFERENCES tools (id) ON DELETE SET NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    -- phase/step, when set, must belong to this finding's run. ON DELETE CASCADE
    -- (a NOT NULL run_id component cannot SET NULL) is safe under the domain rule
    -- that phases and steps are never deleted directly -- only as part of deleting
    -- their run, which already cascades findings through run_id. The same rule
    -- backs the CASCADE on every run-scoped composite FK in this schema.
    CONSTRAINT fk_run_findings_phase FOREIGN KEY (run_id, phase_id)
        REFERENCES run_phases (run_id, id) ON DELETE CASCADE,
    CONSTRAINT fk_run_findings_step FOREIGN KEY (run_id, plan_step_id)
        REFERENCES plan_steps (run_id, id) ON DELETE CASCADE
);

-- payload shape varies by event_type: an EER specialization mapped to a
-- single-table-with-discriminator (the standard mapping for thin, streamed subtypes).
CREATE TABLE agent_events (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    sequence_number bigint NOT NULL,
    event_type      text NOT NULL,                      -- state | error | finished | planning_started | tool_started | tool_finished
    payload         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_events_run_sequence UNIQUE (run_id, sequence_number)
);

-- session_id lives on agent_threads.trace_session_id (it is the thread id);
-- storing it per run was a two-hop transitive dependency.
CREATE TABLE trace_refs (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id     uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    provider   text NOT NULL DEFAULT 'langfuse',
    trace_id   text NOT NULL,
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_trace_refs_provider_trace UNIQUE (provider, trace_id)
);

-- Anchored on plan_step_id, not tool_call_id: the human gate fires before the
-- tool runs, so at insert time there is no call to point at -- and a denial
-- means there never will be one. tool_call_id is backfilled after an approved
-- call executes. The resolution CHECK below binds status to the resolution
-- fields (pending <-> resolved_at AND resolved_by_actor_id both NULL;
-- approved/denied/expired <-> both set) and a partial unique index enforces
-- "one pending per step".
CREATE TABLE approval_requests (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id               uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    plan_step_id         uuid NOT NULL,
    tool_call_id         uuid,                                                  -- backfilled
    status               text NOT NULL DEFAULT 'pending',                       -- pending | approved | denied | expired
    reason               text,
    resolved_by_actor_id uuid REFERENCES actors (id) ON DELETE RESTRICT,
    decision_note        text,
    requested_at         timestamptz NOT NULL DEFAULT now(),
    expires_at           timestamptz,
    resolved_at          timestamptz,
    CONSTRAINT approval_status_valid CHECK (status IN ('pending', 'approved', 'denied', 'expired')),
    CONSTRAINT approval_resolution_consistent CHECK (
        (status = 'pending'
            AND resolved_at IS NULL AND resolved_by_actor_id IS NULL) OR
        (status IN ('approved', 'denied', 'expired')
            AND resolved_at IS NOT NULL AND resolved_by_actor_id IS NOT NULL)
    ),
    -- The gated step, and the backfilled tool call, must both belong to this
    -- approval's run -- otherwise an approval raised for run A could point at a
    -- step or call from run B and silently authorise the wrong execution.
    CONSTRAINT fk_approval_step FOREIGN KEY (run_id, plan_step_id)
        REFERENCES plan_steps (run_id, id) ON DELETE CASCADE,
    CONSTRAINT fk_approval_tool_call FOREIGN KEY (run_id, tool_call_id)
        REFERENCES tool_calls (run_id, id) ON DELETE CASCADE
);

-- Only one live (pending) approval may exist per step; resolved rows are history.
CREATE UNIQUE INDEX uq_one_pending_approval_per_step
    ON approval_requests (plan_step_id) WHERE status = 'pending';

-- Each executed tool call is backfilled onto exactly the one approval that
-- authorised it, so tool_call_id is unique when present (matches the ERD's
-- TOOL_CALLS o|--o| APPROVAL_REQUESTS). Partial: rows are NULL pre-backfill.
CREATE UNIQUE INDEX uq_approval_requests_tool_call
    ON approval_requests (tool_call_id) WHERE tool_call_id IS NOT NULL;

-- A step can be verified more than once: on a RETRY/ANOTHER_TOOL verdict the
-- verifier re-runs the same step and re-checks it, so verifications are
-- attempt-numbered (like tool_calls) rather than one-per-step. Keeping the
-- history append-only is what lets "attempt 1 = not_verified, attempt 2 =
-- verified" be told apart from two competing records; UNIQUE(plan_step_id,
-- attempt) prevents a duplicate attempt number. (A REPLAN instead produces a new
-- plan_step, so that path is already distinct.)
CREATE TABLE verification_results (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    plan_step_id  uuid NOT NULL,
    tool_call_id  uuid,                                 -- optional evidence pointer
    attempt       integer NOT NULL DEFAULT 1,           -- verification round within the step
    result        verification_verdict NOT NULL,
    reason        text,
    deterministic boolean NOT NULL DEFAULT false,       -- true when decided without an LLM round-trip
    evidence      jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_verification_results_step_attempt UNIQUE (plan_step_id, attempt),
    CONSTRAINT verification_results_attempt_positive CHECK (attempt > 0),
    -- The verified step, and the optional evidence call, must both belong to this
    -- result's run -- otherwise a verdict on run A's step could cite run B's call.
    CONSTRAINT fk_verification_results_step FOREIGN KEY (run_id, plan_step_id)
        REFERENCES plan_steps (run_id, id) ON DELETE CASCADE,
    CONSTRAINT fk_verification_results_call FOREIGN KEY (run_id, tool_call_id)
        REFERENCES tool_calls (run_id, id) ON DELETE CASCADE
);

-- =============================================================================
-- Evaluation
-- =============================================================================

CREATE TABLE evaluation_suites (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL,
    version     text NOT NULL,
    description text,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_evaluation_suites_name_version UNIQUE (name, version)
);

CREATE TABLE evaluation_cases (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    suite_id         uuid NOT NULL REFERENCES evaluation_suites (id) ON DELETE CASCADE,
    case_key         text NOT NULL,                     -- stable slug
    category         text,
    goal             text,
    expected_outcome text,
    metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_evaluation_cases_suite_key UNIQUE (suite_id, case_key),
    CONSTRAINT uq_evaluation_cases_id_suite  UNIQUE (id, suite_id)   -- composite-FK target (pins result.suite = case.suite)
);

-- track lives here (it depends on the eval run alone); results no longer copy it.
CREATE TABLE evaluation_runs (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    suite_id           uuid NOT NULL REFERENCES evaluation_suites (id) ON DELETE CASCADE,
    track              agent_track NOT NULL,
    config_snapshot_id uuid REFERENCES config_snapshots (id) ON DELETE RESTRICT,
    started_at         timestamptz NOT NULL DEFAULT now(),
    finished_at        timestamptz,
    CONSTRAINT uq_evaluation_runs_id_suite UNIQUE (id, suite_id)     -- composite-FK target (pins result.suite = run.suite)
);

-- The one true ternary fact (eval run x case -> which agent run). suite_id is
-- carried (denormalized from both parents) so the two composite FKs below force
-- the eval run and the case to belong to the *same* suite -- a plain pair of
-- single-column FKs would let run(suite A) be paired with case(suite B).
-- agent_run_id stays UNIQUE: each eval case execution produces its own fresh
-- agent run, so a run is scored at most once.
CREATE TABLE evaluation_results (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    evaluation_run_id uuid NOT NULL,
    case_id           uuid NOT NULL,
    suite_id          uuid NOT NULL,                     -- shared by both composite FKs
    agent_run_id      uuid NOT NULL UNIQUE REFERENCES agent_runs (id) ON DELETE RESTRICT,  -- the run that produced this result
    success           boolean NOT NULL,
    failure_reason    text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_evaluation_results_run_case UNIQUE (evaluation_run_id, case_id),
    CONSTRAINT fk_evaluation_results_run FOREIGN KEY (evaluation_run_id, suite_id)
        REFERENCES evaluation_runs (id, suite_id) ON DELETE CASCADE,
    CONSTRAINT fk_evaluation_results_case FOREIGN KEY (case_id, suite_id)
        REFERENCES evaluation_cases (id, suite_id) ON DELETE CASCADE
);

-- Replaces evaluation_results.metrics jsonb (a 1NF-violating repeating group of
-- name/value pairs on the very facts the suite exists to compare).
CREATE TABLE evaluation_scores (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id  uuid NOT NULL REFERENCES evaluation_results (id) ON DELETE CASCADE,
    metric     text NOT NULL,
    value      numeric,
    unit       text,
    comment    text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_evaluation_scores_result_metric UNIQUE (result_id, metric)
);

-- =============================================================================
-- Memory and knowledge (Postgres is authoritative; Qdrant mirrors by id)
-- =============================================================================

-- scope is the disjoint-specialization discriminator: an actor-scoped preference
-- must outlive the thread that produced it, while a summary is thread-local. The
-- CHECK enforces the full disjoint binding an EER "d" circle implies -- exactly
-- the one owner column for the scope is set and the others are NULL:
--   actor  -> owner_actor_id set,  thread_id NULL
--   thread -> thread_id set,       owner_actor_id NULL
--   global -> both NULL
-- (Mermaid cardinality cannot express this; the CHECK is the enforcement point.)
CREATE TABLE memory_items (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    scope            text NOT NULL,                     -- actor | thread | global
    owner_actor_id   uuid REFERENCES actors (id) ON DELETE CASCADE,
    thread_id        text REFERENCES agent_threads (id) ON DELETE CASCADE,
    memory_type      text NOT NULL,                     -- preference | fact | summary | correction
    content          text NOT NULL,
    confidence       numeric,
    superseded_by_id uuid REFERENCES memory_items (id) ON DELETE SET NULL,
    metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    last_accessed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT memory_scope_valid CHECK (scope IN ('actor', 'thread', 'global')),
    CONSTRAINT memory_type_valid  CHECK (memory_type IN ('preference', 'fact', 'summary', 'correction')),
    CONSTRAINT memory_scope_binding CHECK (
        (scope = 'actor'  AND owner_actor_id IS NOT NULL AND thread_id IS NULL) OR
        (scope = 'thread' AND thread_id IS NOT NULL AND owner_actor_id IS NULL) OR
        (scope = 'global' AND owner_actor_id IS NULL AND thread_id IS NULL)
    )
);

CREATE TABLE knowledge_sources (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_uri   text NOT NULL UNIQUE,                   -- the source's identity
    source_type  text NOT NULL,                         -- file | url | repo | doc
    content_hash text NOT NULL,                          -- indexed, not unique: detects a changed source on re-index without forbidding two distinct URIs that share bytes
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    indexed_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_source_type_valid CHECK (source_type IN ('file', 'url', 'repo', 'doc'))
);

CREATE TABLE knowledge_chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   uuid NOT NULL REFERENCES knowledge_sources (id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content     text NOT NULL,
    token_count integer,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    indexed_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_knowledge_chunks_source_index UNIQUE (source_id, chunk_index)
);

-- =============================================================================
-- Indexes on foreign keys (skipped where a UNIQUE constraint already leads with
-- the column, e.g. tools(server_id,name), evaluation_scores(result_id,metric)).
-- run_id is indexed everywhere: it is the partition key for the hottest query.
-- =============================================================================
CREATE INDEX ix_agent_threads_owner        ON agent_threads (owner_actor_id);
CREATE INDEX ix_agent_tasks_thread         ON agent_tasks (thread_id);
CREATE INDEX ix_task_attachments_task      ON task_attachments (task_id);
CREATE INDEX ix_agent_runs_task            ON agent_runs (task_id);
CREATE INDEX ix_agent_runs_config          ON agent_runs (config_snapshot_id);
CREATE INDEX ix_run_phases_run             ON run_phases (run_id);
CREATE INDEX ix_plans_run                  ON plans (run_id);
CREATE INDEX ix_plans_phase                ON plans (phase_id);
CREATE INDEX ix_plan_steps_plan            ON plan_steps (plan_id);
CREATE INDEX ix_plan_steps_run             ON plan_steps (run_id);
CREATE INDEX ix_plan_steps_tool            ON plan_steps (tool_id);
CREATE INDEX ix_tool_calls_run             ON tool_calls (run_id);
CREATE INDEX ix_tool_calls_step            ON tool_calls (plan_step_id);
CREATE INDEX ix_tool_calls_tool            ON tool_calls (tool_id);
CREATE INDEX ix_tool_calls_sandbox         ON tool_calls (sandbox_session_id);
CREATE INDEX ix_tool_calls_policy          ON tool_calls (risk_policy_id);
CREATE INDEX ix_llm_calls_run              ON llm_calls (run_id);
CREATE INDEX ix_llm_calls_step             ON llm_calls (plan_step_id);
CREATE INDEX ix_run_findings_run           ON run_findings (run_id);
CREATE INDEX ix_run_findings_phase         ON run_findings (phase_id);
CREATE INDEX ix_run_findings_step          ON run_findings (plan_step_id);
CREATE INDEX ix_run_findings_tool          ON run_findings (source_tool_id);
CREATE INDEX ix_trace_refs_run             ON trace_refs (run_id);
CREATE INDEX ix_sandbox_sessions_run       ON sandbox_sessions (run_id);
CREATE INDEX ix_approval_requests_run      ON approval_requests (run_id);
CREATE INDEX ix_approval_requests_step     ON approval_requests (plan_step_id);
CREATE INDEX ix_approval_requests_call     ON approval_requests (tool_call_id);
CREATE INDEX ix_approval_requests_actor    ON approval_requests (resolved_by_actor_id);
CREATE INDEX ix_verification_results_run   ON verification_results (run_id);
CREATE INDEX ix_verification_results_step  ON verification_results (plan_step_id);
CREATE INDEX ix_verification_results_call  ON verification_results (tool_call_id);
CREATE INDEX ix_tools_server               ON tools (server_id);
CREATE INDEX ix_evaluation_cases_suite     ON evaluation_cases (suite_id);
CREATE INDEX ix_evaluation_runs_suite      ON evaluation_runs (suite_id);
CREATE INDEX ix_evaluation_runs_config     ON evaluation_runs (config_snapshot_id);
CREATE INDEX ix_evaluation_results_case    ON evaluation_results (case_id);
CREATE INDEX ix_memory_items_owner         ON memory_items (owner_actor_id);
CREATE INDEX ix_memory_items_thread        ON memory_items (thread_id);
CREATE INDEX ix_memory_items_superseded    ON memory_items (superseded_by_id);
-- Non-FK lookup: content_hash is how a re-index decides a source changed (not unique).
CREATE INDEX ix_knowledge_sources_hash     ON knowledge_sources (content_hash);

-- =============================================================================
-- updated_at triggers (the only tables carrying updated_at)
-- =============================================================================
CREATE TRIGGER trg_agent_threads_updated_at BEFORE UPDATE ON agent_threads
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_agent_tasks_updated_at BEFORE UPDATE ON agent_tasks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- v_run_metrics - the run counters that used to be denormalized onto agent_runs.
-- Correct by construction. Correlated subqueries (not a 3-way join + GROUP BY)
-- so the three independent child fan-outs cannot multiply each other's counts.
-- Promote to a materialized view or trigger-maintained rollup only if run-list
-- latency ever demands it; the numbers stay derived either way.
-- =============================================================================
CREATE VIEW v_run_metrics AS
SELECT
    r.id      AS run_id,
    r.task_id AS task_id,
    (SELECT count(*) FROM plan_steps ps WHERE ps.run_id = r.id)                       AS steps_taken,
    (SELECT count(*) FROM tool_calls tc WHERE tc.run_id = r.id)                       AS tool_calls,
    (SELECT count(*) FROM llm_calls  lc WHERE lc.run_id = r.id)                       AS llm_calls,
    COALESCE((SELECT sum(lc.total_tokens) FROM llm_calls lc WHERE lc.run_id = r.id), 0) AS total_tokens,
    COALESCE((SELECT sum(lc.cost)         FROM llm_calls lc WHERE lc.run_id = r.id), 0) AS cost
FROM agent_runs r;

COMMIT;

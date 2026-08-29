"""SQL for the Postgres run spine, kept out of the repository logic.

Each statement targets exactly the columns this change writes and leans on the
schema's server-side defaults for everything else (``gen_random_uuid()`` ids,
``now()`` timestamps, the generated ``duration_ms``). Enum-typed columns take a
text value with an explicit ``::enum`` cast; jsonb columns take a value wrapped
in ``psycopg.types.json.Jsonb``.
"""

from __future__ import annotations

# One shared service identity owns API-created threads. external_id is UNIQUE,
# so DO UPDATE (a no-op touch) lets RETURNING yield the id on conflict too.
UPSERT_ACTOR = """
INSERT INTO actors (kind, external_id, display_name)
VALUES ('system', %s, %s)
ON CONFLICT (external_id) DO UPDATE SET display_name = EXCLUDED.display_name
RETURNING id
"""

# thread id is TEXT and equals the LangGraph thread_id we generate.
UPSERT_THREAD = """
INSERT INTO agent_threads (id, owner_actor_id, title)
VALUES (%s, %s, %s)
ON CONFLICT (id) DO UPDATE SET updated_at = now()
RETURNING id
"""

INSERT_TASK = """
INSERT INTO agent_tasks (id, thread_id, goal, track, status, metadata)
VALUES (%s, %s, %s, %s::agent_track, %s::task_status, %s)
ON CONFLICT (id) DO UPDATE SET goal = EXCLUDED.goal
RETURNING id
"""

# Content-addressed: the no-op DO UPDATE makes RETURNING fire on conflict, so a
# repeated config reuses its existing snapshot row.
UPSERT_CONFIG_SNAPSHOT = """
INSERT INTO config_snapshots (
    content_hash, llm_config, execution_config, sandbox_config,
    permissions_config, checkpoint_config, tracing_config,
    behaviour_config, prompts_config
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (content_hash) DO UPDATE SET content_hash = EXCLUDED.content_hash
RETURNING id
"""

# attempt is derived so a retried task (a second run) gets attempt 2 and does
# not collide with UNIQUE (task_id, attempt).
INSERT_RUN = """
INSERT INTO agent_runs (task_id, attempt, config_snapshot_id, status)
VALUES (
    %s,
    (SELECT COALESCE(MAX(attempt), 0) + 1 FROM agent_runs WHERE task_id = %s),
    %s,
    %s::run_status
)
RETURNING id
"""

MARK_RUN_RUNNING = """
UPDATE agent_runs
SET status = 'running'::run_status, started_at = now()
WHERE id = %s
"""

INSERT_EVENT = """
INSERT INTO agent_events (run_id, sequence_number, event_type, payload)
VALUES (%s, %s, %s, %s)
ON CONFLICT (run_id, sequence_number) DO NOTHING
"""

FINALIZE_RUN = """
UPDATE agent_runs
SET status = %s::run_status, output = %s, last_error = %s, finished_at = now()
WHERE id = %s
"""

UPDATE_TASK_STATUS = """
UPDATE agent_tasks SET status = %s::task_status WHERE id = %s
"""

SELECT_TASK = """
SELECT id, thread_id, goal, track, metadata, created_at
FROM agent_tasks
WHERE id = %s
"""

SELECT_LATEST_RUN_STATUS = """
SELECT status FROM agent_runs
WHERE task_id = %s
ORDER BY attempt DESC
LIMIT 1
"""

UPSERT_MCP_SERVER = """
INSERT INTO mcp_servers (name, base_url)
VALUES (%s, %s)
ON CONFLICT (name) DO UPDATE SET base_url = EXCLUDED.base_url, enabled = true
RETURNING id
"""

UPSERT_TOOL = """
INSERT INTO tools (server_id, name, description, input_schema)
VALUES (%s, %s, %s, %s)
ON CONFLICT (server_id, name) DO UPDATE SET
    description = EXCLUDED.description,
    input_schema = EXCLUDED.input_schema,
    last_seen_at = now()
RETURNING id
"""

INSERT_LLM_CALL = """
INSERT INTO llm_calls (
    run_id, node_name, provider, model, prompt_tokens, completion_tokens,
    cost, error, started_at, finished_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

INSERT_TOOL_CALL = """
INSERT INTO tool_calls (
    run_id, tool_id, arguments, success, output, error, risk_level,
    risk_reason, attempt, started_at, finished_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s::risk_level, %s, %s, %s, %s)
"""

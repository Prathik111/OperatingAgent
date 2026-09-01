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

SELECT_THREADS = """
SELECT
    thread.id,
    thread.title,
    COUNT(task.id),
    thread.created_at,
    thread.updated_at
FROM agent_threads AS thread
JOIN actors AS owner ON owner.id = thread.owner_actor_id
LEFT JOIN agent_tasks AS task ON task.thread_id = thread.id
WHERE owner.external_id = %s
GROUP BY thread.id
ORDER BY thread.updated_at DESC, thread.id DESC
LIMIT %s OFFSET %s
"""

SELECT_THREAD_EXISTS = """
SELECT 1
FROM agent_threads AS thread
JOIN actors AS owner ON owner.id = thread.owner_actor_id
WHERE thread.id = %s AND owner.external_id = %s
"""

SELECT_TASKS_BY_THREAD = """
SELECT
    task.id,
    task.thread_id,
    task.goal,
    task.track,
    task.metadata,
    task.created_at,
    latest_run.status
FROM agent_tasks AS task
LEFT JOIN LATERAL (
    SELECT run.status
    FROM agent_runs AS run
    WHERE run.task_id = task.id
    ORDER BY run.attempt DESC
    LIMIT 1
) AS latest_run ON true
WHERE task.thread_id = %s
ORDER BY task.created_at DESC, task.id DESC
LIMIT %s OFFSET %s
"""

SELECT_LATEST_RUN_STATUS = """
SELECT status FROM agent_runs
WHERE task_id = %s
ORDER BY attempt DESC
LIMIT 1
"""

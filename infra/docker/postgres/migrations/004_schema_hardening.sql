BEGIN;

-- Repair the historical representation where id='001_base' and version=1
-- could be stored as two unrelated rows. Normalize orphaned NULL-id row
-- to 001_base when absent, avoiding version=1 unique conflict.
UPDATE schema_migrations SET id = '001_base'
WHERE id IS NULL AND version = 1
  AND NOT EXISTS (SELECT 1 FROM schema_migrations WHERE id = '001_base');
DELETE FROM schema_migrations
WHERE id IS NULL AND version = 1
  AND EXISTS (SELECT 1 FROM schema_migrations WHERE id = '001_base');
UPDATE schema_migrations SET version = 1
WHERE id = '001_base' AND version IS NULL
  AND NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = 1 AND id IS DISTINCT FROM '001_base');
DELETE FROM schema_migrations
WHERE id = '002_import_legacy_agent_native'
  AND EXISTS (
      SELECT 1 FROM schema_migrations
      WHERE id = '003_import_legacy_agent_native'
  );
ALTER TABLE schema_migrations ALTER COLUMN id SET NOT NULL;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_index
        WHERE indrelid = 'schema_migrations'::regclass
          AND indisunique
          AND indkey[0] = (SELECT attnum FROM pg_attribute WHERE attrelid = 'schema_migrations'::regclass AND attname = 'id')
    ) THEN
        ALTER TABLE schema_migrations ADD PRIMARY KEY (id);
    END IF;
END $$;

CREATE OR REPLACE FUNCTION prevent_agent_task_thread_change() RETURNS trigger AS $$
BEGIN
    IF NEW.thread_id IS DISTINCT FROM OLD.thread_id THEN
        RAISE EXCEPTION 'agent_tasks.thread_id is immutable';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_agent_tasks_thread_immutable ON agent_tasks;
CREATE TRIGGER trg_agent_tasks_thread_immutable
    BEFORE UPDATE OF thread_id ON agent_tasks
    FOR EACH ROW EXECUTE FUNCTION prevent_agent_task_thread_change();

DROP INDEX IF EXISTS ix_conversation_messages_thread;

DROP VIEW IF EXISTS v_run_metrics_extended;
DROP VIEW IF EXISTS v_run_metrics;

CREATE VIEW v_run_metrics AS
SELECT
    r.id AS run_id,
    r.task_id AS task_id,
    (SELECT count(*) FROM plan_steps ps WHERE ps.run_id = r.id) AS steps_taken,
    (SELECT count(*) FROM tool_calls tc WHERE tc.run_id = r.id) AS tool_calls,
    (SELECT count(*) FROM llm_calls lc WHERE lc.run_id = r.id) AS llm_calls,
    (SELECT sum(lc.total_tokens) FROM llm_calls lc WHERE lc.run_id = r.id) AS total_tokens,
    (SELECT sum(lc.cost) FROM llm_calls lc WHERE lc.run_id = r.id) AS cost,
    (SELECT count(*) FROM llm_calls lc WHERE lc.run_id = r.id AND
        (lc.prompt_tokens IS NULL OR lc.completion_tokens IS NULL)) AS llm_calls_with_unknown_usage,
    (SELECT count(*) FROM llm_calls lc WHERE lc.run_id = r.id AND lc.cost IS NULL) AS llm_calls_with_unknown_cost
FROM agent_runs r;

CREATE VIEW v_run_metrics_extended AS
SELECT
    r.id AS run_id,
    r.task_id AS task_id,
    at.track,
    at.thread_id,
    r.status,
    r.duration_ms AS latency_ms,
    (SELECT count(*) FROM plan_steps ps WHERE ps.run_id = r.id) AS steps_taken,
    (SELECT count(*) FROM tool_calls tc WHERE tc.run_id = r.id) AS tool_calls,
    (SELECT count(*) FROM tool_calls tc WHERE tc.run_id = r.id AND tc.success = true) AS tool_calls_succeeded,
    (SELECT count(*) FROM tool_calls tc WHERE tc.run_id = r.id AND tc.success = false) AS tool_calls_failed,
    (SELECT count(*) FROM tool_calls tc WHERE tc.run_id = r.id AND tc.success IS NULL) AS tool_calls_unresolved,
    (SELECT count(*) FROM llm_calls lc WHERE lc.run_id = r.id) AS llm_calls,
    (SELECT sum(lc.total_tokens) FROM llm_calls lc WHERE lc.run_id = r.id) AS total_tokens,
    (SELECT sum(lc.cost) FROM llm_calls lc WHERE lc.run_id = r.id) AS cost,
    (SELECT count(*) FROM llm_calls lc WHERE lc.run_id = r.id AND
        (lc.prompt_tokens IS NULL OR lc.completion_tokens IS NULL)) AS llm_calls_with_unknown_usage,
    (SELECT count(*) FROM llm_calls lc WHERE lc.run_id = r.id AND lc.cost IS NULL) AS llm_calls_with_unknown_cost
FROM agent_runs r
JOIN agent_tasks at ON at.id = r.task_id;

INSERT INTO schema_migrations (id, version)
VALUES ('004_schema_hardening', 4)
ON CONFLICT (id) DO NOTHING;

COMMIT;

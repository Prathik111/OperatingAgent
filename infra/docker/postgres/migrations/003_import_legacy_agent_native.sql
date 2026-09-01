-- Import the pre-canonical agent_native_{tasks,runs,plans} tables in place.
-- The source tables are retained; metadata keys make the import idempotent.
BEGIN;

DO $$
DECLARE
    legacy_task record;
    legacy_run record;
    legacy_plan record;
    actor_uuid uuid;
    task_uuid uuid;
    run_uuid uuid;
    config_uuid uuid;
    phase_uuid uuid;
    plan_uuid uuid;
    server_uuid uuid;
    tool_uuid uuid;
    step_uuid uuid;
    step_data jsonb;
    call_index integer;
    prompt_count integer;
    completion_count integer;
BEGIN
    IF to_regclass('public.agent_native_tasks') IS NULL
       OR to_regclass('public.agent_native_runs') IS NULL THEN
        RETURN;
    END IF;

    INSERT INTO actors (kind, external_id, display_name, metadata)
    VALUES ('system', 'legacy-agent-native-import', 'Legacy agent-native import', '{}')
    ON CONFLICT (external_id) DO UPDATE SET display_name = EXCLUDED.display_name
    RETURNING id INTO actor_uuid;

    INSERT INTO config_snapshots (content_hash, behaviour_config)
    VALUES ('legacy-agent-native-v1', '{"source":"agent_native_runs"}')
    ON CONFLICT (content_hash) DO UPDATE SET behaviour_config = EXCLUDED.behaviour_config
    RETURNING id INTO config_uuid;

    INSERT INTO mcp_servers (name, enabled)
    VALUES ('legacy-agent-native', false)
    ON CONFLICT (name) DO UPDATE SET enabled = EXCLUDED.enabled
    RETURNING id INTO server_uuid;

    FOR legacy_task IN SELECT * FROM agent_native_tasks ORDER BY created_at, id LOOP
        INSERT INTO agent_threads (id, owner_actor_id, title, metadata, created_at)
        VALUES (
            legacy_task.thread_id,
            actor_uuid,
            legacy_task.goal,
            jsonb_build_object('legacy_source', 'agent_native_tasks'),
            legacy_task.created_at
        )
        ON CONFLICT (id) DO NOTHING;

        SELECT id INTO task_uuid FROM agent_tasks
        WHERE metadata->>'legacy_task_id' = legacy_task.id;
        IF task_uuid IS NULL THEN
            INSERT INTO agent_tasks (thread_id, goal, track, status, metadata, created_at)
            VALUES (
                legacy_task.thread_id,
                legacy_task.goal,
                'native',
                'planning',
                legacy_task.metadata || jsonb_build_object(
                    'legacy_source', 'agent_native_tasks',
                    'legacy_task_id', legacy_task.id
                ),
                legacy_task.created_at
            )
            RETURNING id INTO task_uuid;
        END IF;

        FOR legacy_run IN SELECT * FROM agent_native_runs WHERE task_id = legacy_task.id LOOP
            SELECT id INTO run_uuid FROM agent_runs
            WHERE metadata->>'legacy_run_id' = legacy_run.id::text;
            IF run_uuid IS NULL THEN
                INSERT INTO agent_runs (
                    task_id, config_snapshot_id, status, output, last_error,
                    metadata, started_at, finished_at
                ) VALUES (
                    task_uuid,
                    config_uuid,
                    CASE legacy_run.status
                        WHEN 'completed' THEN 'completed'::run_status
                        WHEN 'failed' THEN 'failed'::run_status
                        WHEN 'interrupted' THEN 'interrupted'::run_status
                        ELSE 'created'::run_status
                    END,
                    legacy_run.output,
                    legacy_run.failure_reason,
                    legacy_run.metadata || jsonb_build_object(
                        'legacy_source', 'agent_native_runs',
                        'legacy_run_id', legacy_run.id,
                        'replans', legacy_run.replans
                    ),
                    legacy_run.finished_at - (legacy_run.duration_ms * interval '1 millisecond'),
                    legacy_run.finished_at
                ) RETURNING id INTO run_uuid;
            END IF;

            UPDATE agent_tasks SET status = CASE
                WHEN legacy_run.status = 'completed' THEN 'completed'::task_status
                WHEN legacy_run.status = 'failed' THEN 'failed'::task_status
                WHEN legacy_run.status = 'interrupted' THEN 'interrupted'::task_status
                ELSE status
            END WHERE id = task_uuid;

            IF NOT EXISTS (SELECT 1 FROM run_phases WHERE run_id = run_uuid) THEN
                INSERT INTO run_phases (run_id, sequence, phase, entry_reason, entered_at, exited_at)
                VALUES (run_uuid, 0, 'complete', 'Imported legacy plan', legacy_task.created_at, legacy_run.finished_at)
                RETURNING id INTO phase_uuid;
            ELSE
                SELECT id INTO phase_uuid FROM run_phases WHERE run_id = run_uuid ORDER BY sequence LIMIT 1;
            END IF;

            legacy_plan := NULL;
            IF to_regclass('public.agent_native_plans') IS NOT NULL THEN
                SELECT * INTO legacy_plan FROM agent_native_plans WHERE task_id = legacy_task.id LIMIT 1;
            END IF;
            IF legacy_plan.task_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM plans WHERE run_id = run_uuid) THEN
                INSERT INTO plans (run_id, phase_id, revision, summary, reasoning, created_at)
                VALUES (run_uuid, phase_uuid, 0, 'Imported legacy plan', 'Imported from agent_native_plans', legacy_plan.created_at)
                RETURNING id INTO plan_uuid;

                FOR step_data IN SELECT value FROM jsonb_array_elements(legacy_plan.steps) LOOP
                    tool_uuid := NULL;
                    IF COALESCE(step_data->>'tool_name', '') <> '' THEN
                        INSERT INTO tools (server_id, name, description, last_seen_at)
                        VALUES (server_uuid, step_data->>'tool_name', 'Imported legacy tool', legacy_run.finished_at)
                        ON CONFLICT (server_id, name) DO UPDATE SET last_seen_at = EXCLUDED.last_seen_at
                        RETURNING id INTO tool_uuid;
                    END IF;
                    INSERT INTO plan_steps (
                        plan_id, run_id, step_number, description, tool_id,
                        arguments, status, output, started_at, finished_at
                    ) VALUES (
                        plan_uuid,
                        run_uuid,
                        COALESCE((step_data->>'step_number')::integer, 0),
                        step_data->>'description',
                        tool_uuid,
                        jsonb_build_object('legacy_check', step_data->>'check', 'legacy_step_id', step_data->>'id'),
                        CASE WHEN step_data->>'status' IN ('created','pending','running','completed','failed','interrupted')
                             THEN (step_data->>'status')::run_status ELSE 'created'::run_status END,
                        step_data->>'result',
                        legacy_task.created_at,
                        legacy_run.finished_at
                    ) RETURNING id INTO step_uuid;
                END LOOP;
            END IF;

            IF NOT EXISTS (SELECT 1 FROM llm_calls WHERE run_id = run_uuid) THEN
                FOR call_index IN 1..COALESCE(legacy_run.llm_calls, 0) LOOP
                    prompt_count := CASE WHEN call_index = legacy_run.llm_calls
                        THEN legacy_run.total_tokens
                             - (legacy_run.total_tokens / GREATEST(legacy_run.llm_calls, 1))
                               * (legacy_run.llm_calls - 1)
                        ELSE legacy_run.total_tokens / GREATEST(legacy_run.llm_calls, 1) END;
                    completion_count := 0;
                    INSERT INTO llm_calls (run_id, node_name, prompt_tokens, completion_tokens, cost, started_at, finished_at)
                    VALUES (run_uuid, 'legacy_import', prompt_count, completion_count,
                            legacy_run.cost / GREATEST(legacy_run.llm_calls, 1),
                            legacy_task.created_at, legacy_run.finished_at);
                END LOOP;
            END IF;

            SELECT id INTO tool_uuid FROM tools WHERE server_id = server_uuid ORDER BY discovered_at LIMIT 1;
            IF tool_uuid IS NULL THEN
                INSERT INTO tools (server_id, name, description)
                VALUES (server_uuid, 'legacy_unknown_tool', 'Imported aggregate tool call')
                RETURNING id INTO tool_uuid;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM tool_calls WHERE run_id = run_uuid) THEN
                FOR call_index IN 1..COALESCE(legacy_run.tool_calls, 0) LOOP
                    INSERT INTO tool_calls (run_id, tool_id, attempt, success, output, started_at, finished_at)
                    VALUES (run_uuid, tool_uuid, call_index, true,
                            jsonb_build_object('legacy_import', true),
                            legacy_task.created_at, legacy_run.finished_at);
                END LOOP;
            END IF;
        END LOOP;
    END LOOP;
END $$;

INSERT INTO schema_migrations (id, version)
VALUES ('003_import_legacy_agent_native', 3)
ON CONFLICT (id) DO NOTHING;

COMMIT;

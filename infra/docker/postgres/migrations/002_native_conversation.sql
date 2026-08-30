-- Native-only persistence layered on the canonical cross-track schema.
BEGIN;

CREATE OR REPLACE FUNCTION check_conversation_message_thread() RETURNS trigger AS $$
BEGIN
    IF NEW.run_id IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM agent_runs ar
        JOIN agent_tasks at ON at.id = ar.task_id
        WHERE ar.id = NEW.run_id AND at.thread_id = NEW.thread_id
    ) THEN
        RAISE EXCEPTION 'conversation_messages.run_id % does not belong to thread %',
            NEW.run_id, NEW.thread_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS conversation_messages (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id  text NOT NULL REFERENCES agent_threads (id) ON DELETE CASCADE,
    ordinal    bigint NOT NULL GENERATED ALWAYS AS IDENTITY,
    -- Conversation history is thread-owned. Deleting a run preserves the turn
    -- while explicitly dropping only its provenance link.
    run_id     uuid REFERENCES agent_runs (id) ON DELETE SET NULL,
    role       text NOT NULL CHECK (role IN ('system','user','assistant','tool')),
    parts      jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(parts) = 'array'),
    model      text NOT NULL DEFAULT '',
    usage      jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_conversation_messages_thread_ordinal UNIQUE (thread_id, ordinal)
);

DROP TRIGGER IF EXISTS trg_conversation_messages_thread_check ON conversation_messages;
CREATE TRIGGER trg_conversation_messages_thread_check
    BEFORE INSERT OR UPDATE ON conversation_messages
    FOR EACH ROW EXECUTE FUNCTION check_conversation_message_thread();

CREATE TABLE IF NOT EXISTS native_event_sequences (
    thread_id      text PRIMARY KEY REFERENCES agent_threads (id) ON DELETE CASCADE,
    last_sequence bigint NOT NULL DEFAULT 0 CHECK (last_sequence >= 0)
);

CREATE TABLE IF NOT EXISTS native_permission_grants (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_pattern     text NOT NULL,
    duration         text NOT NULL CHECK (duration IN ('once','session','always')),
    thread_id        text REFERENCES agent_threads (id) ON DELETE CASCADE,
    argument_pattern text NOT NULL DEFAULT '',
    consumed_at      timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT native_perm_scope_binding CHECK (
        (duration IN ('once','session') AND thread_id IS NOT NULL) OR
        (duration = 'always' AND thread_id IS NULL)
    ),
    CONSTRAINT native_perm_consumed_scope CHECK (
        consumed_at IS NULL OR duration = 'once'
    )
);

-- UNIQUE(thread_id, ordinal) already serves thread-prefix conversation reads.
CREATE INDEX IF NOT EXISTS ix_conversation_messages_run ON conversation_messages (run_id);
CREATE INDEX IF NOT EXISTS ix_native_grants_thread ON native_permission_grants (thread_id);

INSERT INTO schema_migrations (id, version)
VALUES ('002_native_conversation', 2)
ON CONFLICT (id) DO NOTHING;

COMMIT;

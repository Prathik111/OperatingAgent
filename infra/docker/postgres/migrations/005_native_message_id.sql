-- Preserve original string message ids across Postgres round-trips.
BEGIN;
ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS native_message_id text;
CREATE INDEX IF NOT EXISTS ix_conversation_messages_native_id ON conversation_messages (native_message_id);
INSERT INTO schema_migrations (id, version)
VALUES ('005_native_message_id', 5)
ON CONFLICT (id) DO NOTHING;
COMMIT;

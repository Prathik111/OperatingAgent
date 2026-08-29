-- agent-native: the whole schema, in one file.
--
-- Six tables, one per thing the agent needs to remember: the sessions it has had,
-- the messages in them, the numbered events those produced, a receipt per run,
-- the permissions the user granted, and the notes it was asked to keep.
--
-- Two things here are load-bearing rather than decorative:
--
--   * `sessions.event_sequence` is the event counter. It lives on the session row
--     so that handing out the next number is one atomic UPDATE ... RETURNING -
--     two runs on the same session can ask at the same moment and cannot get the
--     same number. A separate counter table would need a lock; a Python-side
--     counter would be wrong the first time two processes shared a database.
--
--   * `events (session_id, sequence)` is the primary key, so a duplicate number
--     is rejected by the database rather than quietly stored. Replay depends on
--     those numbers being unique and gapless-in-order, and this is what enforces
--     it.
--
-- Safe to run more than once.

CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    agent             TEXT        NOT NULL DEFAULT 'build',
    title             TEXT        NOT NULL DEFAULT '',
    working_directory TEXT        NOT NULL DEFAULT '.',
    revision          INTEGER     NOT NULL DEFAULT 0,
    event_sequence    BIGINT      NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- `ordinal` exists so messages can be read back in the order they were written.
-- Sorting by created_at alone breaks the moment two messages share a timestamp,
-- and a conversation in the wrong order is a conversation the model misreads.
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    ordinal    BIGSERIAL,
    session_id TEXT        NOT NULL,
    role       TEXT        NOT NULL,
    parts      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    model      TEXT        NOT NULL DEFAULT '',
    usage      JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_session_ordinal_idx
    ON messages (session_id, ordinal);

CREATE TABLE IF NOT EXISTS events (
    session_id TEXT        NOT NULL,
    sequence   BIGINT      NOT NULL,
    type       TEXT        NOT NULL,
    run_id     TEXT        NOT NULL DEFAULT '',
    data       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    time       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, sequence)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    session_id       TEXT        NOT NULL,
    status           TEXT        NOT NULL,
    turns            INTEGER     NOT NULL DEFAULT 0,
    input_tokens     INTEGER     NOT NULL DEFAULT 0,
    output_tokens    INTEGER     NOT NULL DEFAULT 0,
    cached_tokens    INTEGER     NOT NULL DEFAULT 0,
    duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_usd         DOUBLE PRECISION NOT NULL DEFAULT 0,
    model            TEXT        NOT NULL DEFAULT '',
    retries          INTEGER     NOT NULL DEFAULT 0,
    error            TEXT        NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS runs_session_idx ON runs (session_id, created_at);

-- An empty session_id means the grant applies everywhere ("always"), which is
-- why this is a plain TEXT default '' and not a nullable foreign key: "no
-- session" is a meaningful value here, not missing data.
--
-- `argument_pattern` is how a yes gets narrowed to a place - "writes under
-- notes/ are fine". Empty means the grant covers the tool whatever it is handed,
-- which is the wider and older meaning, so '' is the right default for a row
-- written before this column existed.
CREATE TABLE IF NOT EXISTS permission_grants (
    id               BIGSERIAL PRIMARY KEY,
    tool_pattern     TEXT        NOT NULL,
    duration         TEXT        NOT NULL,
    session_id       TEXT        NOT NULL DEFAULT '',
    argument_pattern TEXT        NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Added after the table shipped, so an existing database gets the column too.
ALTER TABLE permission_grants
    ADD COLUMN IF NOT EXISTS argument_pattern TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS permission_grants_session_idx
    ON permission_grants (session_id);

-- Notes the agent was asked to keep, across runs.
--
-- There is deliberately no index for searching the text. Matching happens in
-- Python, in `memory.py`, for both stores - which means a query returns the same
-- notes in the same order whether it ran against a dict or against this table.
-- Pushing the scoring into SQL would be faster and would quietly make the two
-- stores disagree, and for a handful of notes per session there is nothing to
-- win. `last_used_at` is what recall bumps, so notes that keep proving useful
-- float to the top of the next session's prompt.
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    session_id   TEXT        NOT NULL DEFAULT '',
    kind         TEXT        NOT NULL DEFAULT 'fact',
    text         TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memories_session_idx ON memories (session_id, last_used_at);

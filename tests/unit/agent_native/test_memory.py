"""Memory: the notes the agent keeps, and the project file the user writes.

All offline. `MemoryStore` is exercised against the *real* in-memory Database, so
these pin the ranking and scoping rules that both stores must obey - no Postgres
and no network in sight. The rules that matter: recall ranks by how much of the
query a note matches, using it floats it up, and a note scoped to another session
is never returned.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_native.database import MemoryDatabase
from agent_native.memory import (
    MAX_INSTRUCTIONS_CHARS,
    Memory,
    MemoryKind,
    MemoryStore,
    read_project_instructions,
)


def _store() -> MemoryStore:
    return MemoryStore(MemoryDatabase())


# -- what a note is ---------------------------------------------------------
async def test_remember_strips_text_and_stores_the_note():
    store = _store()
    note = await store.remember("  use tabs, not spaces  ", MemoryKind.PREFERENCE)
    assert note.text == "use tabs, not spaces"
    assert note.kind == MemoryKind.PREFERENCE
    # It's actually stored: recall by a distinctive word finds exactly it.
    found = await store.recall("tabs")
    assert [m.id for m in found] == [note.id]


async def test_an_unknown_kind_is_filed_as_a_fact_not_refused():
    store = _store()
    note = await store.remember("the port is 8080", kind="nonsense")
    assert note.kind == MemoryKind.FACT


# -- recall ranks by how much of the query a note matches -------------------
async def test_recall_ranks_by_keyword_overlap_and_drops_non_matches():
    store = _store()
    await store.remember("deploy runs on staging then production")   # 1 word overlap
    best = await store.remember("the deploy script lives in scripts")  # 2 words
    await store.remember("lunch is at noon")                          # 0 overlap

    found = await store.recall("where is the deploy script")
    texts = [m.text for m in found]
    assert found[0].text == best.text            # most overlap ranks first
    assert "lunch is at noon" not in texts       # zero overlap is never recalled


async def test_recall_touches_what_it_returns_so_it_floats_up():
    store = _store()
    note = await store.remember("alpha beta gamma")
    created = note.created_at
    found = await store.recall("alpha")          # using it bumps last_used_at
    assert found[0].last_used_at >= created


async def test_recall_scopes_to_this_session_plus_unscoped_notes():
    store = _store()
    await store.remember("global note about caching", session_id="")
    await store.remember("mine about caching", session_id="s1")
    await store.remember("theirs about caching", session_id="s2")

    found = await store.recall("caching", session_id="s1")
    texts = {m.text for m in found}
    assert "global note about caching" in texts  # unscoped is visible everywhere
    assert "mine about caching" in texts
    assert "theirs about caching" not in texts   # another session's note is not


async def test_recent_returns_most_recently_used_first():
    store = _store()
    first = await store.remember("first note")
    await store.remember("second note")
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    await store._db.touch_memory(first.id, later)   # first becomes most-recent

    recent = await store.recent()
    assert recent[0].id == first.id


# -- scoring, in isolation --------------------------------------------------
def test_score_for_ignores_stopwords_and_short_words():
    note = Memory(text="Deploy the CACHE to production")
    # 'the' is a stop word and 'to' is too short; the rest are real words, matched
    # case-insensitively.
    assert note.score_for({"deploy", "cache"}) == 2
    assert note.score_for({"the", "to"}) == 0
    assert note.score_for(set()) == 0


# -- the project instructions file the *user* writes ------------------------
def test_project_instructions_are_read_from_agent_md():
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "AGENT.md").write_text("Always run tests with uv.\n", encoding="utf-8")
        assert read_project_instructions(folder) == "Always run tests with uv."


def test_missing_instructions_are_empty_not_an_error():
    with tempfile.TemporaryDirectory() as folder:
        assert read_project_instructions(folder) == ""   # no AGENT.md is the norm
    assert read_project_instructions("") == ""            # no folder at all


def test_overlong_instructions_are_truncated():
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "AGENT.md").write_text("x" * (MAX_INSTRUCTIONS_CHARS + 500), encoding="utf-8")
        out = read_project_instructions(folder)
        assert out.endswith("[instructions truncated]")
        assert len(out) <= MAX_INSTRUCTIONS_CHARS + len("\n... [instructions truncated]")

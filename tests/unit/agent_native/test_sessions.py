"""Managing stored sessions: list them, fork one, delete one.

Step 14 gives the outside world a way to work with sessions that already exist,
not just start new ones. Three of the four operations need no model at all - they
only read or move rows - so these tests build a provider-free runtime and a
scripted stand-in for the one path (a real exchange) that does need something to
think:

  * fork copies the whole message history - the system prompt included - into a
    brand-new session with fresh message ids, and leaves the source alone;
  * a fork gets its *own* event stream: the log is not copied, so it starts again
    from one, and diverging either side never renumbers or disturbs the other -
    this is the step's headline promise and its own test below;
  * delete removes a session and everything filed under it - messages, events,
    runs, and the session's own grants and notes - while the global ("always" /
    cross-session) grants and notes are deliberately spared;
  * the CLI `sessions` subcommands (list / fork / delete) drive the same service
    the HTTP API does, so the two surfaces can't drift.

Offline by construction: the model is `_scripted.ScriptedProvider`, no network and
no key. Run under pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src:packages/agent-native \
        python3 packages/agent-native/tests/test_sessions.py
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import sys

from agent_native.config import AgentConfig
from agent_native.database import MemoryDatabase
from agent_native.main import _dispatch_sessions, _render_sessions_table
from agent_native.permissions import PermissionDuration, PermissionGrant
from agent_native.service import AgentRuntime, AgentService

from tests._scripted import ScriptedProvider, scripted_registry, text_event


def _service(db: MemoryDatabase, provider: ScriptedProvider) -> AgentService:
    """A service on a provider-free runtime, save for the one scripted model.

    The same shape `test_resume.py` builds: a real `AgentRuntime` (event bus, tool
    manager, memory, loop) wired to a scripted provider so a message can be sent
    without a key, and `AgentService` on top - the object both the CLI and the API
    call.
    """
    runtime = AgentRuntime(
        database=db,
        model_registry=scripted_registry(provider),
        agents=[AgentConfig(name="build", model="scripted-1")],
    )
    return AgentService(runtime)


# ---------------------------------------------------------------------------
# fork: history is copied, the source is untouched
# ---------------------------------------------------------------------------
async def test_fork_copies_history_into_an_independent_session() -> None:
    """The fork begins knowing everything the source knew, under new ids."""
    db = MemoryDatabase()
    service = _service(db, ScriptedProvider([text_event("ok")]))

    source = await service.create_session(agent="build", title="root", working_directory=".")
    await service.send_message(source.id, "the first thing")
    src_conv = await db.load_conversation(source.id)

    fork = await service.fork_session(source.id)

    assert fork.id != source.id
    assert fork.agent == source.agent                     # same agent...
    assert fork.working_directory == source.working_directory
    assert "fork" in fork.title                            # ...and a titled-as-a-fork title

    fork_conv = await db.load_conversation(fork.id)
    # Same conversation, verbatim: roles and text line up one for one.
    assert [m.role for m in fork_conv.messages] == [m.role for m in src_conv.messages]
    assert [m.text() for m in fork_conv.messages] == [m.text() for m in src_conv.messages]
    # But every copied message is a *new* row under the new session.
    assert {m.id for m in fork_conv.messages}.isdisjoint({m.id for m in src_conv.messages})
    assert all(m.session_id == fork.id for m in fork_conv.messages)
    # The source is exactly as it was - forking read it, it didn't move it.
    assert [m.id for m in (await db.load_conversation(source.id)).messages] == [
        m.id for m in src_conv.messages
    ]


async def test_forking_a_missing_session_is_a_clean_error() -> None:
    """Nothing to branch from -> KeyError, which the CLI/API turn into not-found."""
    db = MemoryDatabase()
    service = _service(db, ScriptedProvider([text_event("ok")]))
    raised = False
    try:
        await service.fork_session("no-such-session")
    except KeyError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# The headline promise: forked sessions have independent event streams
# ---------------------------------------------------------------------------
async def test_forked_sessions_have_independent_event_streams() -> None:
    """Fork a session, diverge the two, confirm each has its own event stream.

    The plan's verify for this step, word for word. A fork does not copy the event
    log, so it starts numbering from one; and because each session counts its own
    events, sending a message to one side never advances, renumbers, or reaches into
    the other side's log.
    """
    db = MemoryDatabase()
    service = _service(db, ScriptedProvider([text_event("ok")]))

    source = await service.create_session(agent="build", title="root", working_directory=".")
    await service.send_message(source.id, "first message")
    at_fork = await db.load_events(source.id, 0)
    a_tip = at_fork[-1].sequence
    assert a_tip > 0                                       # the source has a numbered stream

    fork = await service.fork_session(source.id)
    assert await db.load_events(fork.id, 0) == []          # the fork starts with a clean log

    # Diverge down the fork first. The source's stream must not move.
    await service.send_message(fork.id, "down the fork")
    assert (await db.load_events(source.id, 0))[-1].sequence == a_tip
    b_events = await db.load_events(fork.id, 0)
    assert b_events, "the fork got its own events"
    assert [e.sequence for e in b_events] == list(range(1, len(b_events) + 1))  # numbered from 1
    assert {e.session_id for e in b_events} == {fork.id}
    b_tip = b_events[-1].sequence

    # Now diverge down the source. Its stream continues past the fork point, the
    # fork's stays exactly where it was, and neither log holds the other's events.
    await service.send_message(source.id, "down the source")
    a_events = await db.load_events(source.id, 0)
    appended = [e for e in a_events if e.sequence > a_tip]
    assert appended and appended[0].sequence == a_tip + 1  # the source kept climbing
    assert {e.session_id for e in a_events} == {source.id}
    assert (await db.load_events(fork.id, 0))[-1].sequence == b_tip  # the fork was untouched


# ---------------------------------------------------------------------------
# delete: everything under the session goes; the global notes/grants stay
# ---------------------------------------------------------------------------
async def test_delete_removes_a_session_and_its_data_but_spares_global_notes() -> None:
    """Delete takes the session, messages, events, runs, and its own grants/notes.

    The grants and notes with no session id are global - "always" grants and
    cross-session memories - and are deliberately left in place, so deleting one
    conversation can't quietly revoke a standing permission or forget a fact meant
    to outlive any single session.
    """
    db = MemoryDatabase()
    service = _service(db, ScriptedProvider([text_event("ok")]))

    session = await service.create_session(agent="build", title="doomed", working_directory=".")
    await service.send_message(session.id, "leave a trace")

    # A note and a grant scoped to this session, and one of each that is global.
    await service.runtime.memory.remember("scoped note", "fact", session.id)
    await service.runtime.memory.remember("global note", "fact", "")
    await db.save_permission(
        PermissionGrant(tool_pattern="write*", duration=PermissionDuration.SESSION, session_id=session.id)
    )
    await db.save_permission(
        PermissionGrant(tool_pattern="read*", duration=PermissionDuration.ALWAYS, session_id="")
    )

    # Sanity: the session really has data on every table before we delete it.
    assert await db.get_session(session.id) is not None
    assert (await db.load_conversation(session.id)).messages
    assert await db.load_events(session.id, 0)
    assert await db.list_runs(session.id, 0)

    existed = await service.delete_session(session.id)

    assert existed is True
    assert await db.get_session(session.id) is None
    assert (await db.load_conversation(session.id)).messages == []
    assert await db.load_events(session.id, 0) == []
    assert await db.list_runs(session.id, 0) == []
    # The scoped note and grant are gone; the global ones remain.
    remaining_notes = [m.text for m in await db.load_memories("")]
    assert "global note" in remaining_notes
    assert "scoped note" not in remaining_notes
    remaining_grants = [g.tool_pattern for g in await db.load_permissions(session.id)]
    assert "read*" in remaining_grants          # the global grant survived
    assert "write*" not in remaining_grants     # the session-scoped one did not

    # Deleting again reports "nothing there", which the caller turns into not-found.
    assert await service.delete_session(session.id) is False


# ---------------------------------------------------------------------------
# The CLI surface drives the same service, offline
# ---------------------------------------------------------------------------
async def test_sessions_cli_list_fork_delete() -> None:
    """`sessions list|fork|delete` against an open store, no argparse, no Postgres.

    `_dispatch_sessions` is the CLI's body once the store is open; driving it here
    checks the whole command path - the list table, forking a real session, and
    the not-found return code - through the same `AgentService` the API uses.
    """
    db = MemoryDatabase()
    service = _service(db, ScriptedProvider([text_event("ok")]))
    session = await service.create_session(agent="build", title="alpha", working_directory=".")
    await service.send_message(session.id, "hello")

    # list: prints a table naming the session, and exits 0.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = await _dispatch_sessions(
            argparse.Namespace(command="list", dir="", limit=20), db
        )
    listing = out.getvalue()
    assert rc == 0
    assert session.id in listing
    assert "SESSION" in listing and "LAST RUN" in listing

    # fork: makes a new session in the store and exits 0.
    before = {s.id for s in await db.list_sessions()}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = await _dispatch_sessions(
            argparse.Namespace(command="fork", session=session.id, title=""), db
        )
    assert rc == 0
    after = {s.id for s in await db.list_sessions()}
    new_ids = after - before
    assert len(new_ids) == 1                     # exactly one new session appeared
    assert f"Forked {session.id}" in out.getvalue()

    # delete: removes it and exits 0; a second delete is a 1 (not found).
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = await _dispatch_sessions(
            argparse.Namespace(command="delete", session=session.id), db
        )
    assert rc == 0
    assert await db.get_session(session.id) is None

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = await _dispatch_sessions(
            argparse.Namespace(command="delete", session=session.id), db
        )
    assert rc == 1
    assert "No such session" in err.getvalue()


def test_render_sessions_table_folds_the_last_receipt_onto_each_line() -> None:
    """The list table shows each session with its most recent run, or a dash."""

    class _Session:
        def __init__(self, sid: str) -> None:
            self.id = sid
            self.agent = "build"
            self.title = "t"
            self.working_directory = "."

    class _Run:
        status = "finished"
        cost_usd = 0.0
        duration_seconds = 1.25

    table = _render_sessions_table([(_Session("s-with-run"), _Run()), (_Session("s-no-run"), None)])
    lines = table.splitlines()
    assert lines[0].split() == ["SESSION", "AGENT", "TITLE", "DIR", "LAST", "RUN", "COST", "TIME"]
    with_run = next(line for line in lines if "s-with-run" in line)
    no_run = next(line for line in lines if "s-no-run" in line)
    assert "finished" in with_run and "1.25s" in with_run
    assert " - " in f" {no_run} "        # the no-run row shows a dash for its last run


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    async_tests = [
        test_fork_copies_history_into_an_independent_session,
        test_forking_a_missing_session_is_a_clean_error,
        test_forked_sessions_have_independent_event_streams,
        test_delete_removes_a_session_and_its_data_but_spares_global_notes,
        test_sessions_cli_list_fork_delete,
    ]
    sync_tests = [
        test_render_sessions_table_folds_the_last_receipt_onto_each_line,
    ]
    failures: list = []
    for test in async_tests:
        try:
            asyncio.run(test())
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    for test in sync_tests:
        try:
            test()
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - sessions:")
        for line in failures:
            print("  -", line)
        return 1
    total = len(async_tests) + len(sync_tests)
    print(f"PASS - sessions: {total} tests "
          "(fork copies history + independent event streams, delete spares global "
          "notes/grants, CLI list/fork/delete, list table).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

"""Filesystem checkpoints: snapshot the working folder, then rewind to it.

Step 18's promise is a safety net for a wrong or destructive edit: take a snapshot
before a batch of edits, and if the edits go wrong, restore the folder to exactly
what it was - the same files, the same bytes, nothing that was created since. The
event stream can replay the *conversation*; only this can put the *files* back.

The plan's own verify is the spine of this file: make edits, checkpoint, make more,
rewind, and confirm the folder matches byte for byte. Around it sit the pieces that
make the net usable - it removes files created since the checkpoint and brings back
ones deleted since; it refuses to keep its snapshots inside the folder it snapshots
(a restore would then delete its own history); it writes a small manifest so a later
process can reopen the store and rewind; and, wired to the step-16 hooks, it takes
one snapshot before the first mutating tool of a run and leaves a read alone.

Offline by construction: real files in a temp folder, a scripted stand-in model for
the hook path, no network and no key. Run under pytest, or straight without it:
    PYTHONPATH=packages/agent-native/src python3 packages/agent-native/tests/test_checkpoint.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any

from agent_native.checkpoint import (
    AutoCheckpointer,
    CheckpointStore,
    default_base_for,
    install_auto_checkpoint,
)
from agent_native.config import AgentConfig
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    Session,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.hooks import HookManager, HookPoint
from agent_native.loop import AgentLoop, Cancellation, Limits, RunContext, RunStatus
from agent_native.models.base import ModelRegistry
from agent_native.permissions import Decision, PermissionDecision, Policy, PolicyChain
from agent_native.service import AgentRuntime
from agent_native.tools.base import ToolRegistry
from agent_native.tools.manager import ToolManager

from tests._fake_tools import ReadFileTool, WriteFileTool
from tests._scripted import (
    ScriptedProvider,
    call_event,
    scripted_model,
    scripted_registry,
    text_event,
)


# ---------------------------------------------------------------------------
# Small filesystem helpers
# ---------------------------------------------------------------------------
def _write(root: str, rel: str, content: str) -> None:
    """Write text to root/rel, making parent folders as needed."""
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _manifest(root: str) -> dict:
    """Every regular file under root mapped to its bytes - the folder's fingerprint.

    Two folders are equal *byte for byte* exactly when their manifests are equal:
    same set of relative paths, same contents. This is what the plan's verify means
    by "matches byte for byte", made checkable.
    """
    out: dict = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            with open(full, "rb") as handle:
                out[rel] = handle.read()
    return out


# ---------------------------------------------------------------------------
# The store on its own - the plan's verify and its neighbours
# ---------------------------------------------------------------------------
def test_snapshot_edit_rewind_matches_byte_for_byte() -> None:
    # The plan's verify, verbatim: make edits, checkpoint, make more, rewind, and
    # confirm the folder matches byte for byte. The "more" is deliberately every
    # kind of change at once - a file changed, one added, one deleted, a new nested
    # folder - so the rewind has to undo all four to pass.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        _write(work, "a.txt", "alpha")
        _write(work, "sub/b.txt", "beta")
        _write(work, "sub/deep/c.txt", "gamma")
        before = _manifest(work)

        store = CheckpointStore(base_dir=base)
        checkpoint = store.create(work, label="pre-edit")

        _write(work, "a.txt", "ALPHA, rewritten")            # change a file
        _write(work, "new.txt", "brand new")                  # add a file
        os.remove(os.path.join(work, "sub", "b.txt"))         # delete a file
        _write(work, "sub/deep/extra/d.txt", "delta")         # add a nested folder
        assert _manifest(work) != before                      # sanity: really changed

        store.restore(checkpoint)
        assert _manifest(work) == before                      # byte for byte again


def test_rewind_removes_files_created_after_the_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        _write(work, "keep.txt", "keep me")
        store = CheckpointStore(base_dir=base)
        checkpoint = store.create(work)

        _write(work, "after.txt", "made after the snapshot")
        assert os.path.exists(os.path.join(work, "after.txt"))

        store.restore(checkpoint)
        assert not os.path.exists(os.path.join(work, "after.txt"))  # undone
        assert _manifest(work) == {"keep.txt": b"keep me"}


def test_rewind_restores_a_deleted_file() -> None:
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        _write(work, "precious.txt", "do not lose this")
        store = CheckpointStore(base_dir=base)
        checkpoint = store.create(work)

        os.remove(os.path.join(work, "precious.txt"))          # a wrong delete
        assert not os.path.exists(os.path.join(work, "precious.txt"))

        store.restore(checkpoint)
        assert _manifest(work) == {"precious.txt": b"do not lose this"}  # back, same bytes


def test_multiple_checkpoints_rewind_to_each() -> None:
    # Three states of one file, two checkpoints between them; rewinding to either
    # checkpoint lands on that state, so a store holds a history, not just one undo.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        store = CheckpointStore(base_dir=base)
        _write(work, "f.txt", "v1")
        cp_a = store.create(work, label="v1")
        _write(work, "f.txt", "v2")
        cp_b = store.create(work, label="v2")
        _write(work, "f.txt", "v3")

        store.restore(cp_a)
        assert _manifest(work) == {"f.txt": b"v1"}
        store.restore(cp_b)
        assert _manifest(work) == {"f.txt": b"v2"}


def test_list_is_newest_first_with_get_and_latest() -> None:
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        store = CheckpointStore(base_dir=base)
        _write(work, "f.txt", "one")
        first = store.create(work, label="first")
        _write(work, "f.txt", "two")
        second = store.create(work, label="second")

        listing = store.list()
        assert [cp.id for cp in listing] == [second.id, first.id]   # newest first
        assert store.get(first.id) is first
        assert store.get("nope") is None
        assert store.latest() is second


def test_restore_accepts_a_checkpoint_id_string() -> None:
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        store = CheckpointStore(base_dir=base)
        _write(work, "f.txt", "original")
        checkpoint = store.create(work)
        _write(work, "f.txt", "changed")

        restored = store.restore(checkpoint.id)                     # by id, not object
        assert restored.id == checkpoint.id
        assert _manifest(work) == {"f.txt": b"original"}


def test_base_dir_inside_the_working_folder_is_refused() -> None:
    # A snapshot area inside the folder it snapshots is the one thing create() must
    # refuse: a restore mirrors the snapshot onto the folder and would then delete
    # the store's own earlier snapshots. Better a loud error than silent loss.
    with tempfile.TemporaryDirectory() as work:
        store = CheckpointStore(base_dir=os.path.join(work, ".ckpt"))
        raised = False
        try:
            store.create(work)
        except ValueError:
            raised = True
        assert raised, "create() must reject a base_dir inside the working folder"


def test_default_base_is_outside_the_working_folder_and_stable() -> None:
    with tempfile.TemporaryDirectory() as work:
        base = default_base_for(work)
        root = os.path.realpath(work)
        real_base = os.path.realpath(base)
        # Outside the folder, so a snapshot never copies itself and a restore never
        # eats the store - the invariant create() enforces, here by construction.
        assert real_base != root and not real_base.startswith(root + os.sep)
        # Keyed to the path, so a later `rewind` on the same folder finds the store.
        assert default_base_for(work) == base


def test_persisted_index_reopens_with_load_and_rewinds() -> None:
    # The in-memory list dies with its process; the on-disk manifest doesn't. A
    # fresh store loaded from the same base sees the checkpoint and can rewind to
    # it - which is what `agent-native checkpoints rewind` does in a later process.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        _write(work, "x.txt", "one")
        maker = CheckpointStore(base_dir=base)
        checkpoint = maker.create(work, label="first")

        reopened = CheckpointStore.load(base)
        assert [cp.id for cp in reopened.list()] == [checkpoint.id]
        loaded = reopened.get(checkpoint.id)
        assert loaded is not None
        assert loaded.label == "first"
        assert loaded.root == os.path.realpath(work)

        _write(work, "x.txt", "two")
        _write(work, "y.txt", "added since")
        reopened.restore(checkpoint.id)                             # rewind via the reopened store
        assert _manifest(work) == {"x.txt": b"one"}


def test_load_on_an_empty_base_is_a_store_with_no_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as base:
        store = CheckpointStore.load(base)
        assert store.list() == []
        assert store.latest() is None


# ---------------------------------------------------------------------------
# Wired to the step-16 hooks: snapshot before a batch of edits
# ---------------------------------------------------------------------------
class _AllowAll(Policy):
    """Allows everything, so the only thing shaping the run is the checkpoint hook."""

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        return Decision(PermissionDecision.ALLOW, reason="allow-all")


class _MustNotAsk:
    """A prompter that fails if used - the checkpoint path must never prompt."""

    async def ask(self, request: Any, session_id: str) -> bool:
        raise AssertionError("the auto-checkpointer must never prompt the user")


class _ByTurn:
    """Replays a scripted list of events per turn, then finishes.

    A plain scripted provider repeats the same events every turn, which would loop
    forever on a tool call. This lets a test say "write on turn one, write again on
    turn two, then stop" - the shape the coalescing test needs.
    """

    def __init__(self, turns: list) -> None:
        self._turns = turns
        self.calls = 0

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        index = self.calls
        self.calls += 1
        if index < len(self._turns):
            for event in self._turns[index]:
                yield event
        else:
            yield text_event("done")

    def count_tokens(self, messages: list) -> int:
        return 0


def _loop(hooks: HookManager, provider: Any):
    """A real loop wired to the read/write fake tools, allow-all, must-not-ask."""
    db = MemoryDatabase()
    registry = ModelRegistry()
    registry.register_provider("scripted", provider)
    registry.register_model("scripted-1", scripted_model())
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    manager = ToolManager(tools, PolicyChain([_AllowAll()]), _MustNotAsk())
    loop = AgentLoop(registry, tools, manager, ContextManager(), EventBus(db), db, hooks=hooks)
    return loop, db


def _context(session: Session) -> RunContext:
    return RunContext(
        session=session,
        run_id="run_ckpt",                       # no separator: a top-level run
        config=AgentConfig(model="scripted-1"),
        limits=Limits(max_turns=6),
        cancellation=Cancellation(),
    )


async def _run(loop: AgentLoop, db: MemoryDatabase, work: str) -> Any:
    session = Session(agent="build", working_directory=work)
    await db.create_session(session)
    conv = Conversation([system_message("sys"), user_message(session.id, "do the work")])
    return await loop.run(conv, _context(session))


async def test_auto_checkpoint_snapshots_before_a_mutation_and_rewind_undoes_it() -> None:
    # The end-to-end safety net: a run edits the folder, the auto-checkpointer had
    # snapshotted it first, and rewinding to that snapshot removes the run's edit
    # and leaves the pre-run folder byte for byte.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        _write(work, "seed.txt", "original")
        store = CheckpointStore(base_dir=base)
        hooks = HookManager()
        hooks.register(HookPoint.PRE_TOOL, AutoCheckpointer(store).before_tool)

        provider = _ByTurn([[call_event(0, "write_file", '{"path": "out.txt", "content": "hi"}')]])
        loop, db = _loop(hooks, provider)
        result = await _run(loop, db, work)

        assert result.status is RunStatus.FINISHED
        assert os.path.exists(os.path.join(work, "out.txt"))       # the edit happened
        assert len(store.list()) == 1                              # one snapshot, taken first

        store.restore(store.latest())
        assert _manifest(work) == {"seed.txt": b"original"}        # the edit is undone


async def test_auto_checkpoint_ignores_read_only_calls() -> None:
    # A run that only reads mutates nothing, so there's nothing to protect and no
    # snapshot is taken - the net costs nothing when the agent isn't editing.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        _write(work, "seed.txt", "readable")
        store = CheckpointStore(base_dir=base)
        hooks = HookManager()
        hooks.register(HookPoint.PRE_TOOL, AutoCheckpointer(store).before_tool)

        provider = _ByTurn([[call_event(0, "read_file", '{"path": "seed.txt"}')]])
        loop, db = _loop(hooks, provider)
        result = await _run(loop, db, work)

        assert result.status is RunStatus.FINISHED
        assert store.list() == []                                  # a read triggered nothing


async def test_auto_checkpoint_coalesces_one_snapshot_per_run() -> None:
    # Two edits in one run sit behind a *single* checkpoint - "a checkpoint before a
    # batch of edits", not one per edit. Both edits still happen; only the snapshot
    # is taken once, before the first.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as base:
        store = CheckpointStore(base_dir=base)
        hooks = HookManager()
        hooks.register(HookPoint.PRE_TOOL, AutoCheckpointer(store).before_tool)

        provider = _ByTurn(
            [
                [call_event(0, "write_file", '{"path": "one.txt", "content": "1"}')],
                [call_event(0, "write_file", '{"path": "two.txt", "content": "2"}')],
            ]
        )
        loop, db = _loop(hooks, provider)
        result = await _run(loop, db, work)

        assert result.status is RunStatus.FINISHED
        assert os.path.exists(os.path.join(work, "one.txt"))       # both edits ran
        assert os.path.exists(os.path.join(work, "two.txt"))
        assert len(store.list()) == 1                              # but only one snapshot

        # And that one snapshot is from before *either* edit, so rewinding clears both.
        store.restore(store.latest())
        assert _manifest(work) == {}


async def test_install_auto_checkpoint_registers_on_the_runtime() -> None:
    # What main.py does: install onto a real runtime and get back the store. The
    # hook has to land on the runtime's manager (the one the loop shares), or the
    # net would be installed on nothing.
    with tempfile.TemporaryDirectory() as base:
        runtime = AgentRuntime(
            database=MemoryDatabase(),
            model_registry=scripted_registry(ScriptedProvider([text_event("ok")])),
            agents=[AgentConfig(name="build", model="scripted-1")],
        )
        store = install_auto_checkpoint(runtime, base_dir=base)
        assert isinstance(store, CheckpointStore)
        assert runtime.hooks.has(HookPoint.PRE_TOOL)
        assert runtime.loop._hooks is runtime.hooks                # the shared manager


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    sync_tests = [
        test_snapshot_edit_rewind_matches_byte_for_byte,
        test_rewind_removes_files_created_after_the_checkpoint,
        test_rewind_restores_a_deleted_file,
        test_multiple_checkpoints_rewind_to_each,
        test_list_is_newest_first_with_get_and_latest,
        test_restore_accepts_a_checkpoint_id_string,
        test_base_dir_inside_the_working_folder_is_refused,
        test_default_base_is_outside_the_working_folder_and_stable,
        test_persisted_index_reopens_with_load_and_rewinds,
        test_load_on_an_empty_base_is_a_store_with_no_checkpoints,
    ]
    async_tests = [
        test_auto_checkpoint_snapshots_before_a_mutation_and_rewind_undoes_it,
        test_auto_checkpoint_ignores_read_only_calls,
        test_auto_checkpoint_coalesces_one_snapshot_per_run,
        test_install_auto_checkpoint_registers_on_the_runtime,
    ]
    failures: list = []
    for test in sync_tests:
        try:
            test()
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    for test in async_tests:
        try:
            asyncio.run(test())
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - checkpoint:")
        for line in failures:
            print("  -", line)
        return 1
    total = len(sync_tests) + len(async_tests)
    print(
        f"PASS - checkpoint: {total} tests "
        "(store byte-for-byte + rewind x10, auto-checkpoint via hooks x4)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

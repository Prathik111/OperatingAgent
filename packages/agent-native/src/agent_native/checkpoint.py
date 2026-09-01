"""Filesystem checkpoints: a snapshot of the working folder, and a rewind to it.

The numbered event stream (step 12) replays the *conversation* - every message and
tool call, in order. It says nothing about the *files*. If the agent makes a wrong
or destructive edit, replaying the conversation won't bring the old bytes back.
This module is the missing half: a checkpoint is a snapshot of the working folder
taken before a batch of edits, and a rewind restores the folder to a prior
checkpoint, byte for byte.

The default is a plain filesystem snapshot - the folder is copied, file by file,
into a store-owned area *outside* the working folder. A VCS shadow-branch is the
heavier, fancier alternative; it would slot in behind the same `CheckpointStore`
surface, so this stays swappable. What matters is the guarantee: after `restore`,
the working folder matches the snapshot exactly - the same files with the same
bytes, the same directory shape, and nothing that was created since.

`CheckpointStore` is usable on its own (a UI or CLI calls `create`/`restore`), and
`AutoCheckpointer` wires it to the step-16 hooks so a snapshot is taken
automatically before the first mutating tool of a run - one checkpoint in front of
a whole batch of edits, the safety net turned on without the model having to ask.
Registering it is opt-in: with nothing registered the agent behaves exactly as
before (the hook layer's standing promise).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass

from .config import SKIPPED_NAMES
from .hooks import HookContext, HookPoint


# ---------------------------------------------------------------------------
# The snapshot / mirror primitives
# ---------------------------------------------------------------------------
def _iter_files(root: str, skip: str | None = None):
    """Yield the path (relative to root) of every regular file under root.

    Symlinks are skipped, so a snapshot is a deterministic set of real bytes and a
    link can't be used to reach outside the folder. A `skip` subtree (the store's
    own snapshot area, if it ever sits under root) is pruned entirely.
    """
    root = os.path.realpath(root)
    skip = os.path.realpath(skip) if skip else None
    for dirpath, dirnames, filenames in os.walk(root):
        real = os.path.realpath(dirpath)
        if skip and (real == skip or real.startswith(skip + os.sep)):
            dirnames[:] = []
            continue
        # Exclude virtualenvs, caches and other non-project directories
        dirnames[:] = [d for d in dirnames if d not in SKIPPED_NAMES]
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                continue
            yield os.path.relpath(full, root)


def _snapshot_tree(root: str, dest: str, skip: str | None = None) -> int:
    """Copy every regular file under root into dest, keeping the same layout.

    `copy2` carries the file mode and timestamps across, so a restored file looks
    like the original and not merely reads the same. Returns the file count.
    """
    count = 0
    for rel in _iter_files(root, skip=skip):
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
        shutil.copy2(os.path.join(root, rel), target)
        count += 1
    return count


def _mirror(src: str, dst: str) -> None:
    """Make dst an exact copy of src: same dirs, same files+bytes, nothing extra.

    Three passes: recreate every directory and copy every file from the snapshot
    (so changed files are overwritten and deleted ones come back); remove any file
    in the target that the snapshot doesn't have (so files created since the
    checkpoint are undone); then remove any now-stray directory, deepest first.
    """
    src = os.path.realpath(src)
    dst = os.path.realpath(dst)
    src_files: set = set()
    src_dirs: set = set()

    # 1. every dir and file the snapshot has, into the target.
    for dirpath, _dirnames, filenames in os.walk(src):
        rel_dir = os.path.relpath(dirpath, src)
        rel_dir = "" if rel_dir == "." else rel_dir
        src_dirs.add(rel_dir)
        os.makedirs(os.path.join(dst, rel_dir) if rel_dir else dst, exist_ok=True)
        for name in filenames:
            rel = os.path.join(rel_dir, name) if rel_dir else name
            src_files.add(rel)
            shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))

    # 2. any file the target has that the snapshot doesn't - created since, so drop it.
    for dirpath, _dirnames, filenames in os.walk(dst):
        rel_dir = os.path.relpath(dirpath, dst)
        rel_dir = "" if rel_dir == "." else rel_dir
        for name in filenames:
            rel = os.path.join(rel_dir, name) if rel_dir else name
            full = os.path.join(dst, rel)
            if rel not in src_files:
                # Remove regular files and symlinks (os.remove on a symlink removes the link itself)
                if os.path.islink(full) or os.path.lexists(full):
                    try:
                        os.remove(full)
                    except FileNotFoundError:
                        pass
                continue
            if os.path.islink(full):
                continue

    # 3. any directory the target has that the snapshot doesn't - deepest first, so
    #    a parent is only removed after its (now-empty) children.
    for dirpath, _dirnames, _filenames in os.walk(dst, topdown=False):
        rel_dir = os.path.relpath(dirpath, dst)
        rel_dir = "" if rel_dir == "." else rel_dir
        if rel_dir and rel_dir not in src_dirs:
            try:
                os.rmdir(os.path.join(dst, rel_dir))
            except OSError:
                pass  # not empty (a skipped symlink, say); leave it rather than fail


# ---------------------------------------------------------------------------
# A checkpoint and the store that holds them
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Checkpoint:
    """A single snapshot: which folder, where its copy lives, and when it was taken."""

    id: str
    label: str
    root: str          # realpath of the working folder this snapshots
    path: str          # where the copied tree lives (inside the store's base)
    created_at: float
    file_count: int


class CheckpointStore:
    """Takes and restores snapshots of a working folder.

    Snapshots live under one base directory the store owns. That base must be
    *outside* any folder it snapshots - otherwise a snapshot would copy its own
    earlier snapshots, and a restore would delete them. `create` enforces that.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        self._base = os.path.realpath(base_dir) if base_dir else tempfile.mkdtemp(prefix="agent_ckpt_")
        os.makedirs(self._base, exist_ok=True)
        self._checkpoints: dict = {}  # id -> Checkpoint, in creation order

    @property
    def base_dir(self) -> str:
        return self._base

    def create(self, root: str, label: str = "") -> Checkpoint:
        """Snapshot `root` now and return a handle to it."""
        root = os.path.realpath(root)
        if self._base == root or self._base.startswith(root + os.sep):
            raise ValueError(
                "checkpoint base_dir must be outside the working folder it snapshots; "
                f"base {self._base!r} is inside root {root!r}"
            )
        cid = "ckpt_" + uuid.uuid4().hex[:8]
        dest = os.path.join(self._base, cid)
        os.makedirs(dest, exist_ok=True)
        count = _snapshot_tree(root, dest, skip=self._base)
        checkpoint = Checkpoint(cid, label, root, dest, time.time(), count)
        self._checkpoints[cid] = checkpoint
        self._save_index()  # so a later process can find and restore this one
        return checkpoint

    def restore(self, checkpoint: Checkpoint | str) -> Checkpoint:
        """Rewind the working folder to a checkpoint. Returns the checkpoint restored."""
        cp = checkpoint if isinstance(checkpoint, Checkpoint) else self.get(checkpoint)
        if cp is None:
            raise KeyError(f"No such checkpoint: {checkpoint!r}")
        _mirror(cp.path, cp.root)
        return cp

    def get(self, checkpoint_id: str) -> Checkpoint | None:
        return self._checkpoints.get(checkpoint_id)

    def list(self) -> list:
        """Every checkpoint, newest first."""
        return list(self._checkpoints.values())[::-1]

    def latest(self) -> Checkpoint | None:
        """The most recent checkpoint, or None if there are none."""
        newest = self.list()
        return newest[0] if newest else None

    def clear(self) -> None:
        """Forget every checkpoint and delete their stored copies."""
        shutil.rmtree(self._base, ignore_errors=True)
        os.makedirs(self._base, exist_ok=True)
        self._checkpoints.clear()

    # -- persistence: the store describes itself on disk --------------------
    @property
    def _index_path(self) -> str:
        return os.path.join(self._base, "index.json")

    def _save_index(self) -> None:
        """Write the checkpoint list beside the snapshots, in creation order.

        Snapshots are already on disk; this is the small manifest that says which
        ones exist and what each one snapshots, so `load` can rebuild the store in a
        fresh process (the in-memory dict is gone with the process that made it).
        """
        payload = [asdict(cp) for cp in self._checkpoints.values()]
        directory = os.path.dirname(self._index_path)
        fd, temp_path = tempfile.mkstemp(prefix=".index-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_path, self._index_path)
        except Exception:
            try: os.unlink(temp_path)
            except FileNotFoundError: pass
            raise

    @classmethod
    def load(cls, base_dir: str) -> CheckpointStore:
        """Reopen a store from a base directory a previous run wrote.

        Reads the manifest if present (an empty or missing one just means no
        checkpoints yet), so `agent-native checkpoints rewind` in a later process
        can restore what the run recorded.
        """
        store = cls(base_dir=base_dir)
        try:
            with open(store._index_path, encoding="utf-8") as handle:
                rows = json.load(handle)
        except (FileNotFoundError, ValueError):
            rows = []
        for row in rows:
            cp = Checkpoint(
                id=row["id"],
                label=row.get("label", ""),
                root=row["root"],
                path=row["path"],
                created_at=row.get("created_at", 0.0),
                file_count=row.get("file_count", 0),
            )
            store._checkpoints[cp.id] = cp
        return store


def default_base_for(working_directory: str) -> str:
    """A stable snapshot area for a working folder, guaranteed to sit outside it.

    Keyed to the folder's absolute path so two runs on the same folder share one
    store (and a later `rewind` finds it), and kept under the user's home rather
    than inside the folder - a snapshot area inside the folder it snapshots is the
    one thing `create` refuses, since a restore would then delete its own history.
    """
    resolved = os.path.realpath(working_directory)
    slug = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return os.path.join(os.path.expanduser("~"), ".agent_native", "checkpoints", slug)


# ---------------------------------------------------------------------------
# The step-16 wiring: snapshot before a batch of edits
# ---------------------------------------------------------------------------
class AutoCheckpointer:
    """Snapshots the working folder before the first mutating tool of each run.

    Registered as a PRE_TOOL hook. A read-only call is left alone; the first
    mutating call in a run triggers one snapshot, and later mutations in the same
    run don't - so a whole batch of edits sits behind a single checkpoint, which is
    what "checkpoint before a batch of edits" means. It never vetoes: it returns
    None so the tool proceeds. The snapshot is the safety net, not a gate.
    """

    def __init__(self, store: CheckpointStore) -> None:
        self._store = store
        self._checkpointed: set = set()  # run ids already snapshotted this batch

    async def before_tool(self, context: HookContext) -> None:
        # Only mutations matter, and only once per run. `working_directory` is empty
        # for a run with no folder (nothing to snapshot), so skip that too.
        if context.read_only or not context.working_directory:
            return
        if context.run_id in self._checkpointed:
            return
        self._checkpointed.add(context.run_id)
        await asyncio.to_thread(
            self._store.create, context.working_directory, f"before edits in {context.run_id}"
        )
        return


def install_auto_checkpoint(runtime: object, base_dir: str | None = None) -> CheckpointStore:
    """Turn on automatic pre-edit checkpoints for a runtime, and hand back the store.

    Opt-in on purpose: a runtime with this not installed keeps the exact behaviour
    it had before checkpoints existed. The returned store is what a UI or CLI calls
    to `list` the checkpoints and `restore` one.
    """
    store = CheckpointStore(base_dir=base_dir)
    auto = AutoCheckpointer(store)
    runtime.hooks.register(HookPoint.PRE_TOOL, auto.before_tool)  # type: ignore[attr-defined]
    return store

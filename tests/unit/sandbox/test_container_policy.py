"""Hermetic sandbox policy tests (P0-8, P0-9): no Docker needed.

Covers the ``docker run`` argv the pool builds (isolation flags), the
pool's configurable policy surface, and the timeout path: a hung command
must trigger container destruction (the only primitive that provably stops
an in-container process tree), not just a dead client.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sandbox import (
    DEFAULT_CPUS,
    DEFAULT_IMAGE,
    DEFAULT_MEMORY,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_TMPFS,
    ContainerPool,
    ContainerRunner,
)


def _argv(pool: ContainerPool, **overrides) -> list[str]:
    if overrides:
        pool = ContainerPool(
            image=pool.image,
            memory=pool.memory,
            cpus=pool.cpus,
            network=pool.network,
            pids_limit=overrides.get("pids_limit", pool.pids_limit),
            readonly=overrides.get("readonly", pool.readonly),
            tmpfs=overrides.get("tmpfs", pool.tmpfs),
            cap_drop=overrides.get("cap_drop", pool.cap_drop),
            no_new_privileges=overrides.get("no_new_privileges", pool.no_new_privileges),
            user=overrides.get("user", pool.user),
        )
    return pool._run_args("operating-agent-test", Path("/work"))


def test_defaults_match_documented_policy() -> None:
    pool = ContainerPool()
    assert (pool.image, pool.memory, pool.cpus) == (DEFAULT_IMAGE, DEFAULT_MEMORY, DEFAULT_CPUS)
    assert pool.pids_limit == DEFAULT_PIDS_LIMIT
    assert pool.tmpfs == DEFAULT_TMPFS
    assert pool.readonly is True
    assert pool.cap_drop == ("ALL",)
    assert pool.no_new_privileges is True
    assert pool.user == ""
    assert pool.network is False


def test_default_argv_hardens_the_container() -> None:
    argv = _argv(ContainerPool())
    assert "--cap-drop" in argv and "ALL" in argv
    assert "--security-opt" in argv and "no-new-privileges" in argv
    assert "--pids-limit" in argv and DEFAULT_PIDS_LIMIT in argv
    assert "--read-only" in argv
    assert "--tmpfs" in argv and next(iter(DEFAULT_TMPFS)) in argv
    assert "--network" in argv and "none" in argv
    assert "--memory" in argv and "--cpus" in argv and "--init" in argv
    # The workspace bind mount stays writable despite --read-only.
    mount = argv[argv.index("--mount") + 1]
    assert "target=/workspace" in mount and ":ro" not in mount
    assert "user" not in [a.lstrip("-") for a in argv]


def test_network_true_omits_network_none() -> None:
    argv = _argv(ContainerPool(network=True))
    assert "--network" not in argv


def test_readonly_false_omits_readonly_and_tmpfs() -> None:
    argv = _argv(ContainerPool(), readonly=False)
    assert "--read-only" not in argv
    assert "--tmpfs" not in argv


def test_explicit_user_is_passed_through() -> None:
    argv = _argv(ContainerPool(), user="10000")
    assert "--user" in argv
    assert "10000" in argv


def test_resource_limits_are_configurable() -> None:
    argv = _argv(ContainerPool(), pids_limit="64")
    assert "64" in argv
    pool = ContainerPool(memory="1g", cpus="2.0")
    argv = _argv(pool)
    assert "1g" in argv and "2.0" in argv


class _HangingProc:
    """A subprocess that never finishes until killed, then reaps at once."""

    def __init__(self) -> None:
        self.killed = False
        self.returncode: int | None = None

    async def communicate(self):
        if self.killed:
            return (b"", b"")
        await asyncio.sleep(30)
        return (b"out", b"")

    def kill(self) -> None:
        self.killed = True


async def test_timeout_destroys_container_not_just_client(monkeypatch) -> None:
    """Regression (P0-8): timeout must destroy the container.

    Killing the host-side ``docker exec`` client leaves the in-container
    process tree running. The runner therefore invokes its destroy callback,
    and the pool evicts + removes the container.
    """
    proc = _HangingProc()

    async def fake_exec(*args, **kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    destroyed: list = []

    async def on_timeout():
        destroyed.append(True)

    runner = ContainerRunner("cid-1", "img", on_timeout_destroy=on_timeout)
    out = await runner.run("sleep 30", timeout=0.05)

    assert out.timed_out is True
    assert proc.killed is True
    assert destroyed == [True]


async def test_destroy_runner_evicts_only_on_id_match() -> None:
    """A stale timeout must never evict a recreated runner for the same key."""
    pool = ContainerPool()
    calls: list = []

    async def fake_stop(container_id: str) -> None:
        calls.append(container_id)

    pool._stop = fake_stop  # type: ignore[method-assign]
    first = ContainerRunner("cid-old", "img")
    pool._runners["s:/w"] = first

    # Matching id: evicted and removed.
    await pool._destroy_runner("s:/w", "cid-old")
    assert pool._runners == {}
    assert calls == ["cid-old"]

    # Stale id after recreation: the new runner survives, nothing removed.
    pool._runners["s:/w"] = ContainerRunner("cid-new", "img")
    await pool._destroy_runner("s:/w", "cid-old")
    assert pool._runners["s:/w"].container_id == "cid-new"
    assert calls == ["cid-old"]


async def test_stop_all_clears_everything() -> None:
    pool = ContainerPool()
    removed: list = []

    async def fake_stop(container_id: str) -> None:
        removed.append(container_id)

    pool._stop = fake_stop  # type: ignore[method-assign]
    pool._runners["a"] = ContainerRunner("cid-a", "img")
    pool._runners["b"] = ContainerRunner("cid-b", "img")
    await pool.stop_all()
    assert pool._runners == {}
    assert sorted(removed) == ["cid-a", "cid-b"]


@pytest.mark.regression
async def test_timeout_without_destroy_callback_still_reports(monkeypatch) -> None:
    """A runner with no pool attached degrades to client-kill + report."""

    async def fake_exec(*args, **kwargs):
        return _HangingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    runner = ContainerRunner("cid-1", "img")
    out = await runner.run("sleep 30", timeout=0.05)
    assert out.timed_out is True
    assert out.exit_code == -1

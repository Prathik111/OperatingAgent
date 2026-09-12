"""Live Docker sandbox tests (P0-8, P0-9): timeouts truly stop workloads.

Skipped without a Docker daemon and the test image (``SANDBOX_TEST_IMAGE``,
default ``python:3.12-slim``). These prove, against a real container runtime,
what the hermetic suite can only assert structurally: a timed-out command's
whole process tree dies, repeated timeouts accumulate no containers, the
hardening flags are enforced by the daemon, and the read-only root still
allows workspace writes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from sandbox import ContainerPool

TEST_IMAGE = os.getenv("SANDBOX_TEST_IMAGE", "python:3.12-slim")


def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=15,
            check=False,
        )
        if info.returncode != 0:
            return False
        inspect = subprocess.run(
            ["docker", "image", "inspect", TEST_IMAGE],
            capture_output=True,
            timeout=15,
            check=False,
        )
        return inspect.returncode == 0
    except Exception:  # noqa: BLE001 - any failure means "no usable docker"
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _docker_ok(), reason="needs a Docker daemon and the test image"),
]


def _run_docker(*args: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, timeout=timeout, text=True, check=False
    )


@pytest.fixture
async def pool():
    """A pool on the test image, drained afterwards no matter what."""
    owned = ContainerPool(image=TEST_IMAGE)
    try:
        yield owned
    finally:
        await owned.stop_all()


async def _container_gone(container_id: str) -> bool:
    result = _run_docker("inspect", container_id)
    return result.returncode != 0


async def test_sleep_timeout_destroys_container(pool, tmp_path) -> None:
    """Regression (P0-8): timeout kills the workload AND its container."""
    runner = await pool.get("sleep-timeout", str(tmp_path))
    assert runner is not None
    container_id = runner.container_id

    out = await runner.run("sleep 30", timeout=3)

    assert out.timed_out is True
    assert await _container_gone(container_id)
    assert pool._runners == {}

    # The pool recovers: the next command recreates the container.
    runner2 = await pool.get("sleep-timeout", str(tmp_path))
    assert runner2 is not None
    assert runner2.container_id != container_id
    probe = await runner2.run("echo alive", timeout=10)
    assert probe.exit_code == 0 and "alive" in probe.stdout


async def test_child_processes_do_not_survive_timeout(pool, tmp_path) -> None:
    """Regression (P0-8): grandchildren die with the container.

    ``sleep 60 & wait`` leaves a child the shell merely waits on — killing
    only the client or the parent would orphan it. Container destruction
    tears down the whole PID namespace, so afterwards no container (and
    therefore no member of the tree) can exist.
    """
    runner = await pool.get("child-timeout", str(tmp_path))
    assert runner is not None
    container_id = runner.container_id

    out = await runner.run("sleep 60 & wait", timeout=3)

    assert out.timed_out is True
    assert await _container_gone(container_id)


async def test_repeated_timeouts_leave_no_containers(pool, tmp_path) -> None:
    """Regression (P0-8): timeout destroy + recreate is stable and leak-free."""
    seen: set[str] = set()
    for _ in range(3):
        runner = await pool.get("repeat-timeout", str(tmp_path))
        assert runner is not None
        seen.add(runner.container_id)
        out = await runner.run("sleep 30", timeout=2)
        assert out.timed_out is True
        assert await _container_gone(runner.container_id)
    assert len(seen) == 3
    listed = _run_docker("ps", "-a", "--filter", "name=operating-agent-", "--format", "{{.ID}}")
    survivors = {line.strip() for line in listed.stdout.splitlines() if line.strip()}
    assert not (survivors & seen)


async def test_hardening_flags_are_enforced_live(pool, tmp_path) -> None:
    """Regression (P0-9): the daemon reports the isolation we asked for."""
    runner = await pool.get("hardening", str(tmp_path))
    assert runner is not None
    raw = _run_docker("inspect", runner.container_id)
    assert raw.returncode == 0
    host_config = json.loads(raw.stdout)[0]["HostConfig"]
    assert "ALL" in host_config["CapDrop"]
    assert "no-new-privileges" in " ".join(host_config["SecurityOpt"])
    assert host_config["PidsLimit"] == 256
    assert host_config["ReadonlyRootfs"] is True
    tmpfs = host_config["Tmpfs"] or {}
    tmpfs_paths = tmpfs.keys() if isinstance(tmpfs, dict) else tmpfs
    assert "/tmp" in tmpfs_paths
    assert "none" in host_config["NetworkMode"]
    assert str(host_config["Memory"]) == str(512 * 1024 * 1024)


async def test_workspace_writable_but_root_readonly(pool, tmp_path) -> None:
    """Regression (P0-9): --read-only must not break legitimate agent work,
    while writes outside the workspace fail."""
    runner = await pool.get("readonly", str(tmp_path))
    assert runner is not None

    ok = await runner.run("echo hi > /workspace/note.txt && cat /workspace/note.txt", timeout=15)
    assert ok.exit_code == 0
    assert "hi" in ok.stdout
    assert (tmp_path / "note.txt").read_text().strip() == "hi"

    evil = await runner.run("touch /evil", timeout=15)
    assert evil.exit_code != 0

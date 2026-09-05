"""Shared Docker sandbox lifecycle and failure behavior."""

from __future__ import annotations

import asyncio

from sandbox import ContainerPool


class _Process:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self.stdout, self.stderr


async def test_probe_and_status_report_docker_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("sandbox.shutil.which", lambda _name: None)
    pool = ContainerPool()

    assert await pool.probe() is False
    assert pool.status_line() == "sandbox: off - Docker CLI is not installed"
    await pool.close()


async def test_probe_rejects_a_missing_image(monkeypatch) -> None:
    async def create_process(*args, **_kwargs):
        if args[:2] == ("docker", "info"):
            return _Process(stdout=b"28.0\n")
        return _Process(stderr=b"No such image", returncode=1)

    monkeypatch.setattr("sandbox.shutil.which", lambda _name: "docker")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    pool = ContainerPool(image="missing:image")

    assert await pool.probe() is False
    assert "No such image" in pool.status_line()


async def test_invalid_workspace_does_not_start_a_container(monkeypatch, tmp_path) -> None:
    called = False

    async def create_process(*_args, **_kwargs):
        nonlocal called
        called = True
        return _Process(stdout=b"container-id\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    pool = ContainerPool()

    assert await pool.get("session", str(tmp_path / "missing")) is None
    assert not called
    assert "workspace does not exist" in pool.reason


async def test_get_mounts_workspace_and_close_releases_container(monkeypatch, tmp_path) -> None:
    calls: list[tuple] = []

    async def create_process(*args, **_kwargs):
        calls.append(args)
        if args[:2] == ("docker", "info"):
            return _Process(stdout=b"27.0\n")
        if args[:2] == ("docker", "run"):
            return _Process(stdout=b"container-id\n")
        return _Process()

    monkeypatch.setattr("sandbox.shutil.which", lambda _name: "docker")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    pool = ContainerPool()

    runner = await pool.get("session", str(tmp_path))
    assert runner is not None
    run_call = next(call for call in calls if call[:2] == ("docker", "run"))
    assert "--network" in run_call and "none" in run_call
    mount = next(value for value in run_call if value.startswith("type=bind,"))
    assert f"source={tmp_path.resolve()}" in mount
    assert "target=/workspace" in mount
    assert pool.status_line().startswith("sandbox: on - Docker container")

    await pool.close()
    assert any(call[:3] == ("docker", "rm", "-f") for call in calls)
    await pool.close()

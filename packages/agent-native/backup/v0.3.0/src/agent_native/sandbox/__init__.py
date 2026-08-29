"""Per-task Docker sandbox: container lifecycle + deny-all network egress.

Decision #6: egress defaults to deny (--network none unless the config allows
egress). The allowlist lives in the config file and can be extended via
AGENT_NATIVE_SANDBOX_ALLOWED_HOSTS; app-level host gating happens in
RiskClassifier (config.is_host_allowed), this module enforces at the
container level.

Docker CLI is driven via subprocess (no SDK dependency). All operations are
best-effort and degrade to clear errors, not hangs: every docker invocation
has a timeout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..config import SandboxConfig


@dataclass(slots=True)
class SandboxSession:
    task_id: str
    container_id: str
    workspace_path: Path
    config: SandboxConfig
    created: bool = False


class SandboxError(RuntimeError):
    pass


class SandboxManager:
    """Creates one disposable Docker container per task."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self._sessions: dict[str, SandboxSession] = {}
        self._docker_checked: bool | None = None

    @classmethod
    def docker_available(cls) -> bool:
        docker = shutil.which("docker")
        if docker is None:
            return False
        try:
            proc = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=10,
            )
            return proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def create_session(self, task_id: str, workspace: Path | None = None) -> SandboxSession:
        if not self.config.enabled:
            raise SandboxError("sandbox disabled in config")
        if not self.docker_available():
            raise SandboxError("docker not available - cannot create sandbox")
        if task_id in self._sessions:
            return self._sessions[task_id]

        ws = workspace or (Path.cwd() / self.config.workspace_root / task_id)
        ws.mkdir(parents=True, exist_ok=True)

        name = f"agent-native-{task_id[:24]}-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "create",
            "--name", name,
            "--network", "none" if self.config.egress.lower() == "deny" else "bridge",
            "--cpus", self.config.cpu_limit,
            "--memory", self.config.memory_limit,
            "--stop-timeout", str(self.config.time_limit_s),
            "-v", f"{ws}:{_WORKSPACE_CMD}:Z" if _is_selinux() else f"{ws}:{_WORKSPACE_CMD}",
            "--workdir", _WORKSPACE_CMD,
            "--entrypoint", "sleep",
            self.config.image, "infinity",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SandboxError(f"docker create failed: {e}") from e
        if proc.returncode != 0:
            raise SandboxError(f"docker create failed: {proc.stderr.strip()}")
        container_id = proc.stdout.strip().splitlines()[-1]

        started = subprocess.run(["docker", "start", container_id],
                                 capture_output=True, text=True, timeout=120)
        if started.returncode != 0:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=30)
            raise SandboxError(f"docker start failed: {started.stderr.strip()}")

        session = SandboxSession(task_id=task_id, container_id=container_id,
                                 workspace_path=ws, config=self.config, created=True)
        self._sessions[task_id] = session
        return session

    def get_session(self, task_id: str) -> SandboxSession | None:
        return self._sessions.get(task_id)

    def run_command(
        self,
        task_id: str,
        command: list[str] | str,
        cwd: Path | None = None,
        timeout_s: int | None = None,
    ) -> subprocess.CompletedProcess:
        session = self.get_session(task_id)
        if session is None:
            raise SandboxError(f"no sandbox session for task {task_id}")
        cmd = ["docker", "exec"]
        if cwd is not None:
            cmd += ["--workdir", str(cwd)]
        cmd += [session.container_id, *(_as_list(command))]
        limit = timeout_s or self.config.time_limit_s
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            raise SandboxError(f"command exceeded {limit}s time limit") from None
        except OSError as e:
            raise SandboxError(f"docker exec failed: {e}") from e

    def exec_attached(self, task_id: str, command: list[str]) -> subprocess.Popen:
        """Spawn a long-running process inside the task's session container.

        Used by NativeMCPClient to run an MCP server (stdio transport) inside
        the sandbox: same container, same network policy, same limits.
        """
        session = self.get_session(task_id)
        if session is None:
            raise SandboxError(f"no sandbox session for task {task_id}")
        return subprocess.Popen(
            ["docker", "exec", "-i", session.container_id, *command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def destroy_session(self, task_id: str) -> None:
        session = self._sessions.pop(task_id, None)
        if session is None or not session.created:
            return
        try:
            subprocess.run(["docker", "rm", "-f", session.container_id],
                           capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def destroy_all(self) -> None:
        for task_id in list(self._sessions):
            self.destroy_session(task_id)


_WORKSPACE_CMD = "/workspace"


def _as_list(command: list[str] | str) -> list[str]:
    if isinstance(command, str):
        return ["sh", "-c", command]
    return list(command)


def _is_selinux() -> bool:
    return os.name == "posix" and Path("/sys/fs/selinux/enforce").exists()

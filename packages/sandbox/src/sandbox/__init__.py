"""Small Docker-backed command sandbox shared by both agent tracks.

The workspace is mounted at /workspace in every container. The default image is
the project-owned build from infra/sandbox-images.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

DEFAULT_IMAGE = "operating-agent-sandbox:py312"
DEFAULT_MEMORY = "512m"
DEFAULT_CPUS = "1.0"


@dataclass(slots=True)
class CommandOutput:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def combined(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


class ContainerRunner:
    def __init__(self, container_id: str, image: str) -> None:
        self.container_id = container_id
        self.image = image

    async def run(self, command: str | list[str], timeout: float) -> CommandOutput:
        args = ["docker", "exec", self.container_id]
        if isinstance(command, str):
            args.extend(["sh", "-lc", command])
        else:
            args.extend(str(part) for part in command)
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            return CommandOutput(-1, timed_out=True)
        return CommandOutput(
            process.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )


class ContainerPool:
    """Create one disposable container per logical session."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
        network: bool = False,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.reason = ""
        self._runners: dict[str, ContainerRunner] = {}

    async def available(self) -> bool:
        if shutil.which("docker") is None:
            self.reason = "Docker CLI is not installed"
            return False
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "info", "--format", "{{.ServerVersion}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        except (OSError, TimeoutError) as exc:
            self.reason = str(exc) or "Docker daemon is unavailable"
            return False
        if process.returncode != 0:
            self.reason = stderr.decode(errors="replace").strip() or "Docker daemon is unavailable"
            return False
        self.reason = ""
        return True

    async def get(self, session_id: str, workspace: str) -> ContainerRunner | None:
        root = Path(workspace).expanduser().resolve()
        key = f"{session_id}:{root}"
        existing = self._runners.get(key)
        if existing is not None:
            return existing
        if not root.is_dir() or not await self.available():
            return None
        name = f"operating-agent-{uuid4().hex[:12]}"
        args = [
            "docker", "run", "-d", "--rm", "--init", "--name", name,
            "--workdir", "/workspace", "--memory", self.memory, "--cpus", self.cpus,
            "--mount", f"type=bind,source={root},target=/workspace",
        ]
        if not self.network:
            args.extend(["--network", "none"])
        args.extend([self.image, "sleep", "infinity"])
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
        except (OSError, TimeoutError) as exc:
            self.reason = str(exc) or "could not start Docker container"
            return None
        if process.returncode != 0:
            self.reason = stderr.decode(errors="replace").strip() or "could not start Docker container"
            return None
        runner = ContainerRunner(stdout.decode(errors="replace").strip(), self.image)
        self._runners[key] = runner
        return runner

    async def run(
        self,
        tool: object,
        arguments: dict,
        context: object,
        timeout: float,
    ) -> object | None:
        """Run a native SANDBOX tool in the container for its session workspace."""
        command_builder = getattr(tool, "sandbox_command", None)
        if not callable(command_builder):
            return None
        command = command_builder(arguments)
        if isinstance(command, str):
            sandbox_command: str | list[str] = command
        elif isinstance(command, list) and all(
            isinstance(part, str) for part in command
        ):
            sandbox_command = command
        else:
            return None
        session = getattr(context, "session", None)
        session_id = str(getattr(session, "id", "native"))
        workspace = str(getattr(session, "working_directory", ".") or ".")
        runner = await self.get(session_id, workspace)
        if runner is None:
            return None
        result = await runner.run(sandbox_command, timeout=timeout)
        try:
            from agent_native.tools.base import ToolResult

            output = result.combined()
            if result.timed_out:
                return ToolResult(False, error="command timed out in sandbox")
            if result.exit_code != 0:
                return ToolResult(
                    False,
                    output=output,
                    error=f"command failed in sandbox (exit {result.exit_code})",
                )
            return ToolResult(True, output=output or "(no output)")
        except ImportError:
            return None

    async def stop_all(self) -> None:
        runners, self._runners = self._runners, {}
        await asyncio.gather(
            *(self._stop(runner.container_id) for runner in runners.values()),
            return_exceptions=True,
        )

    async def _stop(self, container_id: str) -> None:
        process = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.communicate()


ContainerSandbox = ContainerPool


def main() -> None:
    print("sandbox package: Docker-backed command isolation")


__all__ = [
    "DEFAULT_CPUS",
    "DEFAULT_IMAGE",
    "DEFAULT_MEMORY",
    "CommandOutput",
    "ContainerPool",
    "ContainerRunner",
    "ContainerSandbox",
]

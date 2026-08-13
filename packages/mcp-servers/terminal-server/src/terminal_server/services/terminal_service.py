"""Terminal service implementation for the terminal server package.

The service keeps all process interaction in one place so the tool layer can
remain thin and reusable.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

#: Environment variable holding a comma-separated executable allowlist.
ALLOWLIST_ENV_VAR = "TERMINAL_SERVER_ALLOWED_COMMANDS"

#: Conservative default allowlist: inspection commands only. Anything that
#: deletes, escalates, reaches the network, or opens a nested shell is absent
#: on purpose. Override with ``TERMINAL_SERVER_ALLOWED_COMMANDS``.
DEFAULT_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "cat",
        "dir",
        "echo",
        "git",
        "grep",
        "head",
        "ls",
        "node",
        "npm",
        "ping",
        "pwd",
        "python",
        "python3",
        "tail",
        "tree",
        "type",
        "uv",
        "wc",
        "where",
        "which",
    }
)

#: Characters that let a caller chain or redirect into a second command.
_SHELL_METACHARACTERS = frozenset({"&", "|", ";", ">", "<", "`", "\n", "\r", "$("})


def _load_allowlist() -> frozenset[str]:
    """Read the executable allowlist from the environment, or use the default."""

    configured = os.environ.get(ALLOWLIST_ENV_VAR)
    if configured is None:
        return DEFAULT_ALLOWED_COMMANDS
    entries = {entry.strip().lower() for entry in configured.split(",") if entry.strip()}
    return frozenset(entries)


def _executable_name(argument: str) -> str:
    """Reduce an argv[0] to a bare, comparable executable name."""

    name = Path(argument).name.lower()
    root, extension = os.path.splitext(name)
    return root if extension in {".exe", ".bat", ".cmd", ".com"} else name


class TerminalService:
    """Small service layer for command execution and process inspection.

    Command execution is gated by an executable allowlist. This is an
    application-level authorization boundary, not a sandbox: it constrains
    *which* programs may run, but approved programs still run with the
    server's own privileges.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        allowed_commands: frozenset[str] | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.allowed_commands = allowed_commands if allowed_commands is not None else _load_allowlist()

    def _authorize(self, command: str) -> list[str]:
        """Parse a command and authorize it against the allowlist.

        Returns:
            The parsed argument vector for the approved command.

        Raises:
            ValueError: If the command is empty or contains shell metacharacters.
            PermissionError: If the executable is not on the allowlist.
        """

        if any(token in command for token in _SHELL_METACHARACTERS):
            raise PermissionError("command chaining and redirection are not permitted.")

        argv = shlex.split(command, posix=sys.platform != "win32")
        if not argv:
            raise ValueError("command must be provided.")

        executable = _executable_name(argv[0])
        if executable not in self.allowed_commands:
            self.logger.warning("command_rejected executable=%s", executable)
            raise PermissionError(
                f"command '{executable}' is not allowed. "
                f"Set {ALLOWLIST_ENV_VAR} to authorize additional executables."
            )
        return argv

    def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 60.0,
        context: Any = None,
    ) -> dict[str, Any]:
        """Run an allowlisted command and return its output payload.

        Raises:
            PermissionError: If the command is not authorized by the allowlist.
        """

        argv = self._authorize(command)
        working_directory = Path(cwd).expanduser().resolve() if cwd else None
        try:
            result = subprocess.run(
                argv,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            if sys.platform != "win32":
                raise RuntimeError(f"command not found: {command}") from None
            # Windows shell builtins (dir, type, ...) need cmd. Pass the parsed
            # argv - never the raw string - so cmd cannot re-parse metacharacters.
            result = subprocess.run(
                ["cmd", "/c", *argv],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            payload = {
                "command": command,
                "cwd": str(working_directory) if working_directory else None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "returncode": None,
                "timed_out": True,
            }
            if context is not None:
                context.logger.info("command_timed_out", command=command, timeout=timeout)
            return payload
        payload = {
            "command": command,
            "cwd": str(working_directory) if working_directory else None,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": False,
        }
        if context is not None:
            context.logger.info("command_executed", command=command, returncode=result.returncode)
        return payload

    def list_processes(self, *, context: Any = None) -> dict[str, Any]:
        """Return a snapshot of currently running processes."""

        if sys.platform == "win32":
            output = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if output.returncode != 0:
                raise RuntimeError(output.stderr.strip() or "tasklist failed")
            processes: list[dict[str, Any]] = []
            for line in output.stdout.splitlines():
                fields = line.strip().strip('"').split('","')
                if len(fields) < 2:
                    continue
                processes.append({"pid": fields[1], "name": fields[0]})
        else:
            output = subprocess.run(
                ["ps", "-e", "-o", "pid=,comm="],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if output.returncode != 0:
                raise RuntimeError(output.stderr.strip() or "ps failed")
            processes = [
                {"pid": line.split()[0], "name": line.split()[1]}
                for line in output.stdout.splitlines()
                if line.strip()
            ]
        payload = {"processes": processes}
        if context is not None:
            context.logger.info("processes_listed", count=len(processes))
        return payload

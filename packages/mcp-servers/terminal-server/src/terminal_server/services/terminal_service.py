"""Terminal service implementation for the terminal server package.

The service keeps all process interaction in one place so the tool layer can
remain thin and reusable.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class TerminalService:
    """Small service layer for command execution and process inspection."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER

    def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 60.0,
        context: Any = None,
    ) -> dict[str, Any]:
        """Run a shell command and return its output payload."""

        working_directory = Path(cwd).expanduser().resolve() if cwd else None
        try:
            result = subprocess.run(
                shlex.split(command),
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
            result = subprocess.run(
                ["cmd", "/c", command],
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

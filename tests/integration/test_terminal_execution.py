"""Integration: ``TerminalService`` actually spawning a process.

The authorization logic is pure and lives in
``tests/unit/mcp_servers/test_terminal_service.py``. What cannot be unit tested
is that the allowlist gate runs *before* a process is created and that a
permitted command's output is captured correctly — those need a real spawn.
"""

from __future__ import annotations

import pytest
from terminal_server.services.terminal_service import TerminalService

ALLOWED = frozenset({"echo", "cat", "pwd", "ls"})


@pytest.fixture
def service() -> TerminalService:
    return TerminalService(allowed_commands=ALLOWED)


def test_run_command_executes_and_captures_output(service: TerminalService) -> None:
    payload = service.run_command("echo hello")
    assert "hello" in payload["stdout"]
    assert payload["returncode"] == 0
    assert payload["timed_out"] is False


@pytest.mark.regression
def test_run_command_rejects_unauthorized_before_spawning(service: TerminalService) -> None:
    """The gate must reject before any process exists — a rejection that happened
    *after* the spawn would already have done the damage."""
    with pytest.raises(PermissionError):
        service.run_command("rm -rf /")

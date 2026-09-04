"""Tests for ``TerminalService`` (terminal-server).

The authorization boundary is the point of this service: a command must parse
cleanly, contain no shell metacharacters (no chaining/redirection), and its
executable must be on the allowlist. Those checks are pure, so they are tested
directly here. Actually spawning a process lives in
``tests/integration/test_terminal_execution.py``.
"""

from __future__ import annotations

import pytest
from terminal_server.services.terminal_service import (
    TerminalService,
    _executable_name,
)

ALLOWED = frozenset({"echo", "cat", "pwd", "ls"})


@pytest.fixture
def service() -> TerminalService:
    return TerminalService(allowed_commands=ALLOWED)


# ---------------------------------------------------------------------------
# _executable_name — normalising argv[0] to a comparable name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("/usr/bin/cat", "cat"),
        ("echo", "echo"),
        ("ECHO.EXE", "echo"),
        ("tool.bat", "tool"),
        ("python3", "python3"),  # non-executable extension is kept
    ],
)
def test_executable_name(argument: str, expected: str) -> None:
    assert _executable_name(argument) == expected


# ---------------------------------------------------------------------------
# _authorize — the allowlist gate
# ---------------------------------------------------------------------------


def test_allowlisted_command_authorizes(service: TerminalService) -> None:
    assert service._authorize("echo hello") == ["echo", "hello"]


def test_path_qualified_executable_is_normalised(service: TerminalService) -> None:
    # A full path whose basename is allowlisted is accepted.
    assert service._authorize("/usr/bin/echo hi")[0].endswith("echo")


@pytest.mark.regression
@pytest.mark.parametrize(
    "command",
    [
        "echo a && echo b",   # &
        "echo a | cat",       # |
        "echo a; echo b",     # ;
        "echo out > file",    # >
        "cat < file",         # <
        "echo `whoami`",      # backtick
        "echo $(whoami)",     # command substitution
    ],
)
def test_shell_metacharacters_are_rejected(service: TerminalService, command: str) -> None:
    """Chaining or redirection would smuggle a non-allowlisted command past the
    gate, so metacharacters are refused outright."""
    with pytest.raises(PermissionError):
        service._authorize(command)


@pytest.mark.parametrize("command", ["", "   "])
def test_empty_command_raises_value_error(service: TerminalService, command: str) -> None:
    with pytest.raises(ValueError):
        service._authorize(command)


@pytest.mark.regression
@pytest.mark.parametrize("command", ["rm -rf /", "curl http://example.com", "sudo reboot"])
def test_non_allowlisted_executable_is_rejected(service: TerminalService, command: str) -> None:
    with pytest.raises(PermissionError):
        service._authorize(command)


# ---------------------------------------------------------------------------
# Allowlist sourcing from the environment
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_default_allowlist_comes_from_environment() -> None:
    """With no explicit allowlist the service reads
    ``TERMINAL_SERVER_ALLOWED_COMMANDS`` (pinned by the test bootstrap). It must
    never default to permitting everything."""
    service = TerminalService()
    assert "echo" in service.allowed_commands
    assert "rm" not in service.allowed_commands

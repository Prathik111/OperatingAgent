"""Tests for ``GitService`` (git-server).

Argument construction — commit-count clamping and the diff target guard that
stops a revision from being parsed as a git flag — is pure, and is tested by
capturing what would be passed to git (``_run`` monkeypatched), so no repository
is needed. Tests against the real ``git`` binary live in
``tests/integration/test_git_repository.py``.
"""

from __future__ import annotations

import pytest
from git_server.services.git_service import GitService

# ---------------------------------------------------------------------------
# Argument construction (pure — _run captured, git never invoked)
# ---------------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch):
    """A GitService whose ``_run`` records its args and returns canned output."""
    service = GitService()
    calls: list[tuple] = []

    def fake_run(repository, *args):
        calls.append(args)
        return "abc123 first\ndef456 second"

    monkeypatch.setattr(service, "_run", fake_run)
    return service, calls


@pytest.mark.regression
def test_log_clamps_excessive_count(captured) -> None:
    """An unbounded count would let a caller pull an arbitrarily large log."""
    service, calls = captured
    service.log(max_count=99_999)
    assert calls[0] == ("log", "--oneline", "-n1000")  # clamped to MAX_LOG_COUNT


@pytest.mark.parametrize(("requested", "flag"), [(0, "-n1"), (-5, "-n1"), (5, "-n5")])
def test_log_count_lower_bound_and_passthrough(captured, requested, flag) -> None:
    service, calls = captured
    service.log(max_count=requested)
    assert calls[0] == ("log", "--oneline", flag)


def test_log_splits_commits_into_lines(captured) -> None:
    service, _ = captured
    result = service.log()
    assert result["commits"] == ["abc123 first", "def456 second"]


@pytest.mark.regression
def test_diff_builds_revision_terminated_args(captured) -> None:
    service, calls = captured
    service.diff(target="HEAD~1")
    # The trailing "--" makes git treat the target as a revision, never a path.
    assert calls[0] == ("diff", "HEAD~1", "--")


@pytest.mark.regression
@pytest.mark.parametrize("target", ["-x", "--upload-pack=evil", "--output=/tmp/x"])
def test_diff_rejects_option_like_target(captured, target) -> None:
    """An option-shaped target would be parsed as a git flag — that is argument
    injection, so it is refused before git is invoked."""
    service, calls = captured
    with pytest.raises(ValueError):
        service.diff(target=target)
    assert calls == []  # rejected before git is ever invoked


def test_status_returns_short_status(captured) -> None:
    service, calls = captured
    result = service.status()
    assert calls[0] == ("status", "--short")
    assert "status" in result

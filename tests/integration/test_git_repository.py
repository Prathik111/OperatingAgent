"""Integration: ``GitService`` against the real ``git`` binary.

Argument construction (count clamping, the diff-target guard) is pure and lives
in ``tests/unit/mcp_servers/test_git_service.py`` with ``_run`` monkeypatched.
These tests drive real git against a throwaway repository, which is the only way
to know the arguments we build are ones git actually accepts.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from git_server.services.git_service import GitService

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary is not installed"
)


@pytest.fixture
def git_repo(tmp_path):
    """A throwaway repo with a single commit."""

    def run(*args):
        subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        )

    run("init")
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    run("add", "a.txt")
    # -c keeps identity local to this invocation — no global git config needed.
    run("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-m", "initial")
    return tmp_path


def test_status_of_clean_repo_is_empty(git_repo) -> None:
    result = GitService().status(str(git_repo))
    assert result["status"] == ""  # nothing outstanding after the commit


def test_status_reports_new_file(git_repo) -> None:
    (git_repo / "new.txt").write_text("x", encoding="utf-8")
    result = GitService().status(str(git_repo))
    assert "new.txt" in result["status"]


def test_log_lists_the_commit(git_repo) -> None:
    result = GitService().log(str(git_repo), max_count=10)
    assert len(result["commits"]) == 1
    assert "initial" in result["commits"][0]


def test_diff_shows_uncommitted_change(git_repo) -> None:
    (git_repo / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    result = GitService().diff(str(git_repo), target="HEAD")
    assert "a.txt" in result["diff"]

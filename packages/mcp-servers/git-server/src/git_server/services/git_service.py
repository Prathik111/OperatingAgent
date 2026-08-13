"""Git service implementation for the git server package.

The service keeps all repository interaction in one place so the tool layer can
remain thin and reusable.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class GitService:
    """Small service layer for Git inspection and diff workflows."""

    #: Upper bound applied to caller-supplied commit counts.
    MAX_LOG_COUNT = 1000

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER

    def _run(self, repository: str, *args: str) -> str:
        repository_path = Path(repository).expanduser().resolve()
        result = subprocess.run(
            ["git", *args],
            cwd=repository_path,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git command failed")
        return result.stdout.strip()

    def status(self, repository: str = ".", *, context: Any = None) -> dict[str, Any]:
        """Return the Git status summary for a repository."""

        payload = {"repository": str(Path(repository).expanduser().resolve()), "status": self._run(repository, "status", "--short")}
        if context is not None:
            context.logger.info("git_status", repository=payload["repository"])
        return payload

    def branches(self, repository: str = ".", *, context: Any = None) -> dict[str, Any]:
        """List the branch names for a repository."""

        payload = {"repository": str(Path(repository).expanduser().resolve()), "branches": self._run(repository, "branch", "--list").splitlines()}
        if context is not None:
            context.logger.info("git_branches", repository=payload["repository"], count=len(payload["branches"]))
        return payload

    def log(self, repository: str = ".", *, max_count: int = 10, context: Any = None) -> dict[str, Any]:
        """Return a short log snapshot for a repository.

        ``max_count`` is clamped to ``[1, MAX_LOG_COUNT]`` so a caller cannot
        request an unbounded history walk.
        """

        bounded_count = max(1, min(int(max_count), self.MAX_LOG_COUNT))
        output = self._run(repository, "log", "--oneline", f"-n{bounded_count}")
        payload = {"repository": str(Path(repository).expanduser().resolve()), "commits": output.splitlines()}
        if context is not None:
            context.logger.info("git_log", repository=payload["repository"], count=len(payload["commits"]))
        return payload

    def diff(self, repository: str = ".", *, target: str = "HEAD", context: Any = None) -> dict[str, Any]:
        """Return the diff text for a repository against a target revision.

        ``target`` is rejected when it looks like an option and is followed by
        ``--`` so Git always treats it as a revision, never as a flag.

        Raises:
            ValueError: If ``target`` starts with ``-``.
        """

        if target.startswith("-"):
            raise ValueError(f"invalid diff target: {target!r}")
        output = self._run(repository, "diff", target, "--")
        payload = {"repository": str(Path(repository).expanduser().resolve()), "diff": output}
        if context is not None:
            context.logger.info("git_diff", repository=payload["repository"], target=target)
        return payload

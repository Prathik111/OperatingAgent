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

    def __init__(self, logger: logging.Logger | None = None, *, root: str | Path | None = None) -> None:
        self.logger = logger or LOGGER
        self.root = Path(root).expanduser().resolve() if root else None
        if self.root is not None and not self.root.is_dir():
            raise ValueError(f"git workspace does not exist: {self.root}")

    def _repository(self, repository: str) -> Path:
        candidate = Path(repository).expanduser()
        if not candidate.is_absolute():
            candidate = (self.root or Path.cwd()) / candidate
        resolved = candidate.resolve()
        if self.root is not None and not resolved.is_relative_to(self.root):
            raise PermissionError(f"git repository escapes workspace: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"git repository does not exist: {resolved}")
        return resolved

    def _run(self, repository: str, *args: str) -> str:
        repository_path = self._repository(repository)
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

        payload = {"repository": str(self._repository(repository)), "status": self._run(repository, "status", "--short")}
        if context is not None:
            context.logger.info("git_status", repository=payload["repository"])
        return payload

    def branches(self, repository: str = ".", *, context: Any = None) -> dict[str, Any]:
        """List the branch names for a repository."""

        payload = {"repository": str(self._repository(repository)), "branches": self._run(repository, "branch", "--list").splitlines()}
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
        payload = {"repository": str(self._repository(repository)), "commits": output.splitlines()}
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
        payload = {"repository": str(self._repository(repository)), "diff": output}
        if context is not None:
            context.logger.info("git_diff", repository=payload["repository"], target=target)
        return payload

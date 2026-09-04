"""Workspace selection and validation shared by API tracks."""

from __future__ import annotations

from pathlib import Path

from .errors import InvalidWorkspace


def resolve_workspace(value: str | None, *, default: str) -> str:
    """Return an absolute existing directory for a task or native session.

    The configured default is allowed to be absent during development; in that
    case the repository working directory is used. Explicit user selections must
    exist so a typo never silently mounts a different directory.
    """
    requested = (value or "").strip()
    candidate = requested or (default or "")
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise InvalidWorkspace(value or candidate) from exc
    if resolved.is_dir():
        return str(resolved)
    if not requested:
        fallback = Path.cwd().resolve()
        if fallback.is_dir():
            return str(fallback)
    raise InvalidWorkspace(value or candidate)

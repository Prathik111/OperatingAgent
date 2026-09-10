"""Production filesystem service for the file-server package.

The service owns all filesystem business logic, while the tool layer is only
responsible for validating input and converting the service payload into a
standardized framework response.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

#: Environment variable that pins the workspace root every path is confined to.
ROOT_ENV_VAR = "FILE_SERVER_ROOT"

#: Chunk size used when streaming file copies, so large files never land in RAM.
COPY_CHUNK_SIZE = 1024 * 1024


def _default_root() -> Path:
    """Resolve the workspace root from the environment, defaulting to the CWD."""

    configured = os.environ.get(ROOT_ENV_VAR)
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


@dataclass(slots=True)
class FileSystemService:
    """Concrete filesystem service that implements reusable file operations.

    Every path is confined to ``root``; requests that escape it are rejected.

    Args:
        logger: Logger for request-scoped diagnostics.
        root: Workspace root that bounds every path this service touches.
    """

    logger: logging.Logger = field(default_factory=lambda: LOGGER)
    root: Path = field(default_factory=_default_root)

    def _normalize_path(self, path: str) -> Path:
        """Resolve a path inside the workspace root.

        Relative paths resolve against ``root``. Absolute paths are accepted
        only when they already point inside ``root``. Anything that escapes -
        via ``..``, a symlink, or an unrelated absolute path - is rejected.

        Raises:
            PermissionError: If the resolved path falls outside ``root``.
        """

        root = self.root.resolve()
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise PermissionError(f"Path escapes the workspace root {root}: {path}")
        return resolved

    def _lexical_rel_parts(self, path: str) -> tuple[str, ...]:
        """Split a user path into parts relative to ``root`` lexically.

        ``..`` and ``.`` are resolved without touching the filesystem, so a
        malicious component cannot hide behind resolution order. Absolute
        paths are normalized lexically first and must land inside ``root``.
        """

        root = self.root.resolve()
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            normalized = Path(os.path.normpath(str(candidate)))
            try:
                rel = normalized.relative_to(root)
            except ValueError:
                raise PermissionError(
                    f"Path escapes the workspace root {root}: {path}"
                ) from None
            parts = rel.parts
        else:
            parts = candidate.parts
        stack: list[str] = []
        for part in parts:
            if part in ("", "."):
                continue
            if part == "..":
                if not stack:
                    raise PermissionError(
                        f"Path escapes the workspace root {root}: {path}"
                    )
                stack.pop()
                continue
            stack.append(part)
        return tuple(stack)

    def _verified_path(self, path: str) -> Path:
        """Resolve ``path`` inside ``root``, rejecting link traversal.

        Every component *as supplied* is lstat-checked, so a symlink is
        caught even when its target stays inside ``root`` (checking only the
        resolved path would lose that information — the resolved path points
        at the target, which is link-free). The chain is then re-resolved and
        required to be stable and confined, which also catches Windows
        junctions (``os.path.islink`` misses those, ``Path.resolve`` follows
        them) and components swapped for a link between check and use.
        Callers re-run this immediately before use (and after any mkdir) to
        narrow the check-then-use window; the residual race needs OS-level
        ``O_NOFOLLOW``/``openat2`` semantics unavailable portably.
        """

        root = self.root.resolve()
        current = root
        for part in self._lexical_rel_parts(path):
            current = current / part
            if os.path.islink(current):
                raise PermissionError(
                    f"Path traverses a symlink inside the workspace root {root}: {path}"
                )
        fresh = current.resolve()
        if fresh != current:
            raise PermissionError(
                f"Path traverses a symlink inside the workspace root {root}: {path}"
            )
        if fresh != root and root not in fresh.parents:
            raise PermissionError(
                f"Path escapes the workspace root {root}: {path}"
            )
        return fresh

    def _ensure_parent_dirs(self, path: str) -> Path:
        """``mkdir -p`` the parent chain of ``path`` without following links.

        Each level is lstat-checked before descent/creation, so a planted
        symlink parent is rejected instead of created through. Returns the
        freshly re-verified target path for immediate use.
        """

        root = self.root.resolve()
        parts = self._lexical_rel_parts(path)
        current = root
        for part in parts[:-1]:
            current = current / part
            if os.path.islink(current):
                raise PermissionError(
                    f"Path traverses a symlink inside the workspace root {root}: {path}"
                )
            if not current.exists():
                try:
                    current.mkdir()
                except FileExistsError:
                    pass
                if os.path.islink(current):
                    raise PermissionError(
                        f"Path traverses a symlink inside the workspace root {root}: {path}"
                    )
                if not current.is_dir():
                    raise NotADirectoryError(f"Not a directory: {current}")
            elif os.path.islink(current) or not current.is_dir():
                if os.path.islink(current):
                    raise PermissionError(
                        f"Path traverses a symlink inside the workspace root {root}: {path}"
                    )
                raise NotADirectoryError(f"Not a directory: {current}")
        return self._verified_path(path)

    def read_file(self, path: str, *, encoding: str = "utf-8", context: Any = None) -> dict[str, Any]:
        """Read a text file and return a serializable payload."""

        resolved = self._verified_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        resolved = self._verified_path(path)
        content = resolved.read_text(encoding=encoding)
        if context is not None:
            context.logger.info("file_read", path=str(resolved))
        return {"path": str(resolved), "content": content, "encoding": encoding}

    def write_file(self, path: str, content: str, *, encoding: str = "utf-8", overwrite: bool = True, context: Any = None) -> dict[str, Any]:
        """Write text content to a file, creating parents as needed."""

        resolved = self._ensure_parent_dirs(path)
        if resolved.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {resolved}")
        resolved = self._verified_path(path)
        resolved.write_text(content, encoding=encoding)
        if context is not None:
            context.logger.info("file_written", path=str(resolved))
        return {"path": str(resolved), "bytes": len(content.encode(encoding)), "encoding": encoding}

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
        encoding: str = "utf-8",
        context: Any = None,
    ) -> dict[str, Any]:
        """Replace an exact text anchor without rewriting unrelated content."""
        resolved = self._verified_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        if not old_string:
            raise ValueError("old_string must be provided and non-empty")
        if old_string == new_string:
            raise ValueError("old_string and new_string are identical")

        resolved = self._verified_path(path)
        bytes_before = resolved.stat().st_size
        content = resolved.read_text(encoding=encoding)
        matches = content.count(old_string)
        if matches == 0:
            raise ValueError("old_string was not found; file was not changed")
        if matches > 1 and not replace_all:
            raise ValueError(
                f"old_string matched in {matches} places; add context or set replace_all"
            )

        replacements = matches if replace_all else 1
        updated = content.replace(old_string, new_string, -1 if replace_all else 1)
        resolved = self._verified_path(path)
        resolved.write_text(updated, encoding=encoding)
        bytes_after = resolved.stat().st_size
        if context is not None:
            context.logger.info(
                "file_edited", path=str(resolved), replacements=replacements
            )
        return {
            "path": str(resolved),
            "replacements": replacements,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "encoding": encoding,
        }

    def delete_file(self, path: str, context: Any = None) -> dict[str, Any]:
        """Delete a file from the filesystem."""

        resolved = self._verified_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        resolved = self._verified_path(path)
        resolved.unlink()
        if context is not None:
            context.logger.info("file_deleted", path=str(resolved))
        return {"path": str(resolved), "deleted": True}

    def copy_file(self, source: str, destination: str, *, overwrite: bool = True, context: Any = None) -> dict[str, Any]:
        """Copy one file to another destination path."""

        source_path = self._verified_path(source)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        destination_path = self._ensure_parent_dirs(destination)
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        if destination_path.exists() and source_path.samefile(destination_path):
            raise PermissionError(
                f"Source and destination refer to the same file: {source_path}"
            )
        source_path = self._verified_path(source)
        destination_path = self._verified_path(destination)
        with source_path.open("rb") as source_file, destination_path.open("wb") as destination_file:
            while chunk := source_file.read(COPY_CHUNK_SIZE):
                destination_file.write(chunk)
        if context is not None:
            context.logger.info("file_copied", source=str(source_path), destination=str(destination_path))
        return {"source": str(source_path), "destination": str(destination_path), "copied": True}

    def move_file(self, source: str, destination: str, *, overwrite: bool = True, context: Any = None) -> dict[str, Any]:
        """Move a file to a new destination path."""

        source_path = self._verified_path(source)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        destination_path = self._ensure_parent_dirs(destination)
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        source_path = self._verified_path(source)
        destination_path = self._verified_path(destination)
        source_path.replace(destination_path)
        if context is not None:
            context.logger.info("file_moved", source=str(source_path), destination=str(destination_path))
        return {"source": str(source_path), "destination": str(destination_path), "moved": True}

    def rename_file(self, source: str, destination: str, context: Any = None) -> dict[str, Any]:
        """Rename a file by moving it to a new file name within the same parent directory."""

        return self.move_file(source, destination, overwrite=True, context=context)

    def list_directory(self, path: str, *, recursive: bool = False, context: Any = None) -> dict[str, Any]:
        """List directory entries and optionally recurse into child directories."""

        resolved = self._verified_path(path)
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"Directory not found: {resolved}")

        def walk(node: Path) -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            for child in sorted(node.iterdir()):
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_dir": child.is_dir(),
                        "is_file": child.is_file(),
                    }
                )
                if recursive and child.is_dir() and not child.is_symlink():
                    entries.extend(walk(child))
            return entries

        items = walk(resolved)
        if context is not None:
            context.logger.info("directory_listed", path=str(resolved), entries=len(items))
        return {"path": str(resolved), "entries": items}

    def create_directory(self, path: str, *, parents: bool = True, context: Any = None) -> dict[str, Any]:
        """Create a directory and any missing parent directories when requested."""

        root = self.root.resolve()
        parts = self._lexical_rel_parts(path)
        current = root
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            current = current / part
            if os.path.islink(current):
                raise PermissionError(
                    f"Path traverses a symlink inside the workspace root {root}: {path}"
                )
            if not current.exists():
                if not last and not parents:
                    raise FileNotFoundError(f"Parent directory not found: {current.parent}")
                try:
                    current.mkdir()
                except FileExistsError:
                    pass
                if os.path.islink(current):
                    raise PermissionError(
                        f"Path traverses a symlink inside the workspace root {root}: {path}"
                    )
            elif last and not current.is_dir():
                raise FileExistsError(f"Path already exists: {current}")
        resolved = self._verified_path(path)
        if context is not None:
            context.logger.info("directory_created", path=str(resolved))
        return {"path": str(resolved), "created": True}

    def delete_directory(self, path: str, *, recursive: bool = False, context: Any = None) -> dict[str, Any]:
        """Delete a directory tree when permitted by the caller."""

        resolved = self._verified_path(path)
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"Directory not found: {resolved}")
        resolved = self._verified_path(path)
        if recursive:
            for root, dirs, files in os.walk(resolved, topdown=False, followlinks=False):
                for file_name in files:
                    Path(root, file_name).unlink(missing_ok=True)
                for dir_name in dirs:
                    dir_path = Path(root, dir_name)
                    if os.path.islink(dir_path):
                        # Never follow a linked dir: remove the link itself.
                        dir_path.unlink(missing_ok=True)
                    else:
                        dir_path.rmdir()
            resolved.rmdir()
        else:
            resolved.rmdir()
        if context is not None:
            context.logger.info("directory_deleted", path=str(resolved), recursive=recursive)
        return {"path": str(resolved), "deleted": True}

    def exists(self, path: str, context: Any = None) -> dict[str, Any]:
        """Return existence and type information for a path."""

        resolved = self._verified_path(path)
        payload = {"path": str(resolved), "exists": resolved.exists(), "is_dir": resolved.is_dir(), "is_file": resolved.is_file()}
        if context is not None:
            context.logger.info("path_checked", path=str(resolved), exists=payload["exists"])
        return payload

    def metadata(self, path: str, context: Any = None) -> dict[str, Any]:
        """Collect reusable metadata for a file or directory path."""

        resolved = self._verified_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        resolved = self._verified_path(path)
        stat = resolved.stat()
        payload = {
            "path": str(resolved),
            "name": resolved.name,
            "parent": str(resolved.parent),
            "size": stat.st_size,
            "is_dir": resolved.is_dir(),
            "is_file": resolved.is_file(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        }
        if context is not None:
            context.logger.info("path_metadata", path=str(resolved))
        return payload

    def search_files(self, root: str, query: str, *, recursive: bool = True, context: Any = None) -> dict[str, Any]:
        """Search a directory tree by file name for a query term."""

        resolved = self._verified_path(root)
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"Directory not found: {resolved}")
        results: list[dict[str, Any]] = []
        for current_root, _, files in os.walk(resolved, followlinks=False):
            for file_name in files:
                if query.lower() in file_name.lower():
                    file_path = Path(current_root, file_name)
                    results.append({"path": str(file_path), "name": file_name})
            if not recursive:
                break
        if context is not None:
            context.logger.info("file_search", root=str(resolved), matches=len(results))
        return {"root": str(resolved), "query": query, "matches": results}

    async def watch_directory(self, path: str, *, interval_seconds: float = 1.0, limit: int = 1, context: Any = None) -> dict[str, Any]:
        """Return a short snapshot of directory changes over a bounded time window."""

        resolved = self._verified_path(path)
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"Directory not found: {resolved}")
        snapshots: list[dict[str, Any]] = []
        for _ in range(limit):
            snapshots.append({"timestamp": datetime.now(UTC).isoformat(), "entries": self.list_directory(str(resolved), recursive=False)["entries"]})
            await asyncio.sleep(interval_seconds)
        if context is not None:
            context.logger.info("directory_watched", path=str(resolved), snapshots=len(snapshots))
        return {"path": str(resolved), "snapshots": snapshots}

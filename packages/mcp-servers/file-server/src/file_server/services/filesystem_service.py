"""Production filesystem service for the file-server package.

The service owns all filesystem business logic, while the tool layer is only
responsible for validating input and converting the service payload into a
standardized framework response.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FileSystemService:
    """Concrete filesystem service that implements reusable file operations.

    Args:
        logger: Logger for request-scoped diagnostics.
    """

    logger: logging.Logger = field(default_factory=lambda: LOGGER)

    def _normalize_path(self, path: str) -> Path:
        """Resolve a path into a normalized absolute ``Path`` object."""

        return Path(path).expanduser().resolve()

    def _ensure_directory(self, path: Path) -> None:
        """Create all missing directory parents for a target file path."""

        path.parent.mkdir(parents=True, exist_ok=True)

    def read_file(self, path: str, *, encoding: str = "utf-8", context: Any = None) -> dict[str, Any]:
        """Read a text file and return a serializable payload."""

        resolved = self._normalize_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        content = resolved.read_text(encoding=encoding)
        if context is not None:
            context.logger.info("file_read", path=str(resolved))
        return {"path": str(resolved), "content": content, "encoding": encoding}

    def write_file(self, path: str, content: str, *, encoding: str = "utf-8", overwrite: bool = True, context: Any = None) -> dict[str, Any]:
        """Write text content to a file, creating parents as needed."""

        resolved = self._normalize_path(path)
        self._ensure_directory(resolved)
        if resolved.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {resolved}")
        resolved.write_text(content, encoding=encoding)
        if context is not None:
            context.logger.info("file_written", path=str(resolved))
        return {"path": str(resolved), "bytes": len(content.encode(encoding)), "encoding": encoding}

    def delete_file(self, path: str, context: Any = None) -> dict[str, Any]:
        """Delete a file from the filesystem."""

        resolved = self._normalize_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        resolved.unlink()
        if context is not None:
            context.logger.info("file_deleted", path=str(resolved))
        return {"path": str(resolved), "deleted": True}

    def copy_file(self, source: str, destination: str, *, overwrite: bool = True, context: Any = None) -> dict[str, Any]:
        """Copy one file to another destination path."""

        source_path = self._normalize_path(source)
        destination_path = self._normalize_path(destination)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        self._ensure_directory(destination_path)
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        destination_path.write_bytes(source_path.read_bytes())
        if context is not None:
            context.logger.info("file_copied", source=str(source_path), destination=str(destination_path))
        return {"source": str(source_path), "destination": str(destination_path), "copied": True}

    def move_file(self, source: str, destination: str, *, overwrite: bool = True, context: Any = None) -> dict[str, Any]:
        """Move a file to a new destination path."""

        source_path = self._normalize_path(source)
        destination_path = self._normalize_path(destination)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        self._ensure_directory(destination_path)
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        source_path.replace(destination_path)
        if context is not None:
            context.logger.info("file_moved", source=str(source_path), destination=str(destination_path))
        return {"source": str(source_path), "destination": str(destination_path), "moved": True}

    def rename_file(self, source: str, destination: str, context: Any = None) -> dict[str, Any]:
        """Rename a file by moving it to a new file name within the same parent directory."""

        return self.move_file(source, destination, overwrite=True, context=context)

    def list_directory(self, path: str, *, recursive: bool = False, context: Any = None) -> dict[str, Any]:
        """List directory entries and optionally recurse into child directories."""

        resolved = self._normalize_path(path)
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
                if recursive and child.is_dir():
                    entries.extend(walk(child))
            return entries

        items = walk(resolved)
        if context is not None:
            context.logger.info("directory_listed", path=str(resolved), entries=len(items))
        return {"path": str(resolved), "entries": items}

    def create_directory(self, path: str, *, parents: bool = True, context: Any = None) -> dict[str, Any]:
        """Create a directory and any missing parent directories when requested."""

        resolved = self._normalize_path(path)
        resolved.mkdir(parents=parents, exist_ok=True)
        if context is not None:
            context.logger.info("directory_created", path=str(resolved))
        return {"path": str(resolved), "created": True}

    def delete_directory(self, path: str, *, recursive: bool = False, context: Any = None) -> dict[str, Any]:
        """Delete a directory tree when permitted by the caller."""

        resolved = self._normalize_path(path)
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"Directory not found: {resolved}")
        if recursive:
            for root, dirs, files in os.walk(resolved, topdown=False):
                for file_name in files:
                    Path(root, file_name).unlink(missing_ok=True)
                for dir_name in dirs:
                    Path(root, dir_name).rmdir()
            resolved.rmdir()
        else:
            resolved.rmdir()
        if context is not None:
            context.logger.info("directory_deleted", path=str(resolved), recursive=recursive)
        return {"path": str(resolved), "deleted": True}

    def exists(self, path: str, context: Any = None) -> dict[str, Any]:
        """Return existence and type information for a path."""

        resolved = self._normalize_path(path)
        payload = {"path": str(resolved), "exists": resolved.exists(), "is_dir": resolved.is_dir(), "is_file": resolved.is_file()}
        if context is not None:
            context.logger.info("path_checked", path=str(resolved), exists=payload["exists"])
        return payload

    def metadata(self, path: str, context: Any = None) -> dict[str, Any]:
        """Collect reusable metadata for a file or directory path."""

        resolved = self._normalize_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        stat = resolved.stat()
        payload = {
            "path": str(resolved),
            "name": resolved.name,
            "parent": str(resolved.parent),
            "size": stat.st_size,
            "is_dir": resolved.is_dir(),
            "is_file": resolved.is_file(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        if context is not None:
            context.logger.info("path_metadata", path=str(resolved))
        return payload

    def search_files(self, root: str, query: str, *, recursive: bool = True, context: Any = None) -> dict[str, Any]:
        """Search a directory tree by file name for a query term."""

        resolved = self._normalize_path(root)
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"Directory not found: {resolved}")
        results: list[dict[str, Any]] = []
        for current_root, _, files in os.walk(resolved):
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

        resolved = self._normalize_path(path)
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"Directory not found: {resolved}")
        snapshots: list[dict[str, Any]] = []
        for _ in range(limit):
            snapshots.append({"timestamp": datetime.now(timezone.utc).isoformat(), "entries": self.list_directory(str(resolved), recursive=False)["entries"]})
            await asyncio.sleep(interval_seconds)
        if context is not None:
            context.logger.info("directory_watched", path=str(resolved), snapshots=len(snapshots))
        return {"path": str(resolved), "snapshots": snapshots}

"""Tests for ``FileSystemService`` (file-server).

The service owns the filesystem business logic and — critically — confines
every path to a workspace root. The security-relevant test is that no path can
escape ``root`` (via ``..``, or an unrelated absolute path); the rest pin the
CRUD surface and its error contract. ``root`` is set to the per-test
``workspace`` dir so tests are isolated and never touch the real repo.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from file_server.services.filesystem_service import FileSystemService


def _try_symlink(target: Path, link: Path, *, target_is_directory: bool = False) -> None:
    """Create ``link -> target``, skipping when the OS forbids symlinks."""
    try:
        if os.path.islink(link) or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")


@pytest.fixture
def service(workspace):
    return FileSystemService(root=workspace)


# Path confinement (the security boundary)


@pytest.mark.regression
def test_relative_parent_escape_is_rejected(service) -> None:
    with pytest.raises(PermissionError):
        service.exists("../outside.txt")


@pytest.mark.regression
def test_absolute_path_outside_root_is_rejected(service, workspace) -> None:
    outside = str(workspace.parent / "sibling.txt")
    with pytest.raises(PermissionError):
        service.exists(outside)


def test_root_itself_is_allowed(service) -> None:
    assert service.exists(".")["exists"] is True


def test_nested_relative_path_inside_root_is_allowed(service) -> None:
    # Confinement rejects escapes, not legitimate descent.
    result = service.write_file("nested/deep/file.txt", "hi")
    assert result["bytes"] == 2


# Read / write


def test_write_then_read_roundtrip(service) -> None:
    service.write_file("note.txt", "hello world")
    payload = service.read_file("note.txt")
    assert payload["content"] == "hello world"
    assert payload["encoding"] == "utf-8"


def test_write_reports_byte_count(service) -> None:
    payload = service.write_file("a.txt", "hello")
    assert payload["bytes"] == 5


@pytest.mark.regression
def test_write_without_overwrite_on_existing_raises(service) -> None:
    service.write_file("a.txt", "first")
    with pytest.raises(FileExistsError):
        service.write_file("a.txt", "second", overwrite=False)


def test_read_missing_file_raises(service) -> None:
    with pytest.raises(FileNotFoundError):
        service.read_file("does-not-exist.txt")


def test_read_directory_as_file_raises(service) -> None:
    service.create_directory("adir")
    with pytest.raises(FileNotFoundError):
        service.read_file("adir")


# Delete / copy / move / rename


def test_delete_file(service) -> None:
    service.write_file("gone.txt", "x")
    service.delete_file("gone.txt")
    assert service.exists("gone.txt")["exists"] is False


def test_delete_missing_file_raises(service) -> None:
    with pytest.raises(FileNotFoundError):
        service.delete_file("nope.txt")


def test_copy_file(service) -> None:
    service.write_file("src.txt", "payload")
    service.copy_file("src.txt", "dst.txt")
    assert service.read_file("dst.txt")["content"] == "payload"
    assert service.exists("src.txt")["exists"] is True  # copy leaves the source


def test_copy_missing_source_raises(service) -> None:
    with pytest.raises(FileNotFoundError):
        service.copy_file("missing.txt", "dst.txt")


@pytest.mark.regression
def test_copy_onto_itself_is_rejected(service) -> None:
    service.write_file("self.txt", "x")
    with pytest.raises(PermissionError):
        service.copy_file("self.txt", "self.txt")


def test_move_file(service) -> None:
    service.write_file("from.txt", "moved")
    service.move_file("from.txt", "to.txt")
    assert service.exists("from.txt")["exists"] is False
    assert service.read_file("to.txt")["content"] == "moved"


def test_rename_file_delegates_to_move(service) -> None:
    service.write_file("old.txt", "data")
    service.rename_file("old.txt", "new.txt")
    assert service.read_file("new.txt")["content"] == "data"


# Directories


def test_list_directory_lists_entries(service) -> None:
    service.write_file("one.txt", "1")
    service.write_file("two.txt", "2")
    names = {e["name"] for e in service.list_directory(".")["entries"]}
    assert {"one.txt", "two.txt"} <= names


def test_list_directory_recursive_descends(service) -> None:
    service.write_file("sub/child.txt", "c")
    names = {e["name"] for e in service.list_directory(".", recursive=True)["entries"]}
    assert {"sub", "child.txt"} <= names


def test_list_missing_directory_raises(service) -> None:
    with pytest.raises(NotADirectoryError):
        service.list_directory("no-such-dir")


def test_create_and_delete_directory(service) -> None:
    service.create_directory("d")
    assert service.exists("d")["is_dir"] is True
    service.delete_directory("d")
    assert service.exists("d")["exists"] is False


def test_delete_nonempty_directory_non_recursive_raises(service) -> None:
    service.write_file("box/item.txt", "x")
    with pytest.raises(OSError):
        service.delete_directory("box", recursive=False)


def test_delete_nonempty_directory_recursive(service) -> None:
    service.write_file("box/item.txt", "x")
    service.delete_directory("box", recursive=True)
    assert service.exists("box")["exists"] is False


# Metadata / existence / search


def test_exists_reports_type(service) -> None:
    service.write_file("f.txt", "x")
    payload = service.exists("f.txt")
    assert payload["exists"] is True
    assert payload["is_file"] is True
    assert payload["is_dir"] is False


def test_metadata_of_file(service) -> None:
    service.write_file("m.txt", "12345")
    meta = service.metadata("m.txt")
    assert meta["name"] == "m.txt"
    assert meta["size"] == 5
    assert meta["is_file"] is True
    assert "modified_at" in meta


def test_metadata_missing_raises(service) -> None:
    with pytest.raises(FileNotFoundError):
        service.metadata("ghost.txt")


def test_search_files_matches_by_name(service) -> None:
    service.write_file("report_final.txt", "x")
    service.write_file("notes.md", "y")
    matches = service.search_files(".", "report")["matches"]
    assert [m["name"] for m in matches] == ["report_final.txt"]


# watch_directory (async)


async def test_watch_directory_returns_snapshots(service) -> None:
    service.write_file("watched.txt", "x")
    result = await service.watch_directory(".", interval_seconds=0.0, limit=1)
    assert len(result["snapshots"]) == 1
    names = {e["name"] for e in result["snapshots"][0]["entries"]}
    assert "watched.txt" in names


# Symlink / TOCTOU hardening (P0-10)
#
# ``_normalize_path`` re-resolves on every call, so a naive outside-symlink is
# already rejected. The gap was subtler: a symlink whose target stays inside
# ``root`` was *followed* (confused deputy), and a directory swapped for a
# symlink between check and use was trusted. The service now rejects any link
# component — pre-placed or swapped in — fail closed.


@pytest.mark.regression
def test_symlink_file_pointing_outside_is_rejected(service, workspace) -> None:
    secret = workspace.parent / f"{workspace.name}_secret.txt"
    secret.write_text("s3cr3t")
    try:
        _try_symlink(secret, workspace / "link.txt")
        with pytest.raises(PermissionError):
            service.read_file("link.txt")
        with pytest.raises(PermissionError):
            service.exists("link.txt")
        with pytest.raises(PermissionError):
            service.metadata("link.txt")
    finally:
        secret.unlink(missing_ok=True)


@pytest.mark.regression
def test_read_through_symlink_dir_inside_root_is_rejected(service, workspace) -> None:
    real = workspace / "real"
    real.mkdir()
    (real / "file.txt").write_text("real-content")
    _try_symlink(real, workspace / "linkdir", target_is_directory=True)
    with pytest.raises(PermissionError):
        service.read_file("linkdir/file.txt")


@pytest.mark.regression
def test_write_through_symlink_dir_is_rejected(service, workspace) -> None:
    real = workspace / "real"
    real.mkdir()
    _try_symlink(real, workspace / "linkdir", target_is_directory=True)
    with pytest.raises(PermissionError):
        service.write_file("linkdir/evil.txt", "x")
    # The link target must not gain a file through the rejected write.
    assert not (real / "evil.txt").exists()


@pytest.mark.regression
def test_dir_swap_with_symlink_is_rejected(service, workspace) -> None:
    service.write_file("a/file.txt", "A")
    service.write_file("b/file.txt", "B")
    shutil.rmtree(workspace / "a")
    _try_symlink(workspace / "b", workspace / "a", target_is_directory=True)
    # Pre-hardening this followed the swapped link and returned "B".
    with pytest.raises(PermissionError):
        service.read_file("a/file.txt")
    with pytest.raises(PermissionError):
        service.write_file("a/new.txt", "x")


@pytest.mark.regression
def test_copy_dest_through_symlink_is_rejected(service, workspace) -> None:
    service.write_file("src.txt", "payload")
    real = workspace / "real"
    real.mkdir()
    _try_symlink(real, workspace / "linkdir", target_is_directory=True)
    with pytest.raises(PermissionError):
        service.copy_file("src.txt", "linkdir/dst.txt")
    assert not (real / "dst.txt").exists()


@pytest.mark.regression
def test_move_dest_through_symlink_is_rejected(service, workspace) -> None:
    service.write_file("src.txt", "payload")
    real = workspace / "real"
    real.mkdir()
    _try_symlink(real, workspace / "linkdir", target_is_directory=True)
    with pytest.raises(PermissionError):
        service.move_file("src.txt", "linkdir/dst.txt")
    # Failed move must leave the source untouched.
    assert service.read_file("src.txt")["content"] == "payload"


@pytest.mark.regression
def test_delete_directory_recursive_unlinks_symlink_without_following(
    service, workspace
) -> None:
    target = workspace / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    box = workspace / "box"
    box.mkdir()
    (box / "other.txt").write_text("other")
    _try_symlink(target, box / "link", target_is_directory=True)
    service.delete_directory("box", recursive=True)
    assert service.exists("box")["exists"] is False
    # Only the link is removed; the link target is preserved.
    assert (target / "keep.txt").read_text() == "keep"


@pytest.mark.regression
def test_list_directory_on_symlink_dir_is_rejected(service, workspace) -> None:
    real = workspace / "real"
    real.mkdir()
    (real / "file.txt").write_text("x")
    _try_symlink(real, workspace / "linkdir", target_is_directory=True)
    with pytest.raises(PermissionError):
        service.list_directory("linkdir")

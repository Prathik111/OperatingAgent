"""Anchor-based surgical file edits: change one region, never the wrong place.

Step 24 adds a surgical alternative to whole-file `write_file`: the caller names
the exact text to find (`old_string`) and what to put there (`new_string`), and
only that span changes. On a large file that is cheaper and far less destructive
than rewriting the whole thing - and it pairs with checkpoints (18), which snapshot
the folder before the edit runs.

The plan's verify is the spine of this file: the tool changes a target region of a
large file *without touching the rest*, and a non-matching anchor *fails safely*
rather than editing the wrong place. Around it sit the other ways an anchor can be
wrong - ambiguous (matches more than once), empty, identical to the replacement -
each of which must refuse and leave the file exactly as it was.

Where the logic lives, and why this reaches across packages: the tool "arrives as
an MCP tool like the file server" (plan step 24), and this repo keeps all business
logic in `FileSystemService` with the tool layer only validating input. `fastmcp`
isn't installed in the offline sandbox, so the FastMCP tool wrapper can't be
imported the normal way - but the service is stdlib-only and fully testable, and
the wrapper is thin enough to exercise by stubbing `fastmcp` and capturing the
function it registers. So this file proves both layers offline:
    * the service (the real anchor logic, the plan's verify), and
    * the wrapper (its input validation forwards to the service).

Run under pytest, or straight without it:
    PYTHONPATH=packages/agent-native/src python3 packages/agent-native/tests/test_edit_file.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

# The file-server package isn't on the default offline PYTHONPATH (the agent
# borrows its tools over MCP, it doesn't import them). Add its src so the
# stdlib-only service is importable, and stub `fastmcp` if it's absent so the
# thin tool wrapper imports too - the wrapper only uses the @mcp.tool decorator
# and a stringized type annotation, neither of which needs the real package.
_FS_SRC = Path(__file__).resolve().parents[2] / "mcp-servers" / "file-server" / "src"
if str(_FS_SRC) not in sys.path:
    sys.path.insert(0, str(_FS_SRC))

try:  # real package on a machine with the extra installed; stub otherwise
    import fastmcp  # noqa: F401
except ImportError:
    # Importing the file_server package runs build_server() at module load, which
    # constructs a FastMCP and registers every tool - so the stub must be a real
    # class that's constructible and offers a .tool decorator in both the bare
    # (@mcp.tool) and called (@mcp.tool(name=...)) forms the server uses. It does
    # nothing with what it's handed; the tests drive the wrapper through their own
    # capturing double, not through this import-time instance.
    class _StubFastMCP:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def tool(self, *args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]  # bare @mcp.tool on a function

            def decorator(fn):
                return fn

            return decorator

        def run(self, *args, **kwargs) -> None:
            pass

    _stub = types.ModuleType("fastmcp")
    _stub.FastMCP = _StubFastMCP
    sys.modules["fastmcp"] = _stub

from file_server.services.filesystem_service import FileSystemService
from file_server.tools.edit_file import register_edit_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _service(root: str) -> FileSystemService:
    return FileSystemService(root=Path(root).resolve())


def _write(root: str, rel: str, content: str) -> str:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _big_file_text(target_line: int = 250, total: int = 500) -> str:
    """A large file where exactly one line is a unique anchor we can aim at."""
    lines = []
    for i in range(total):
        if i == target_line:
            lines.append(f"line {i}: THE ONE UNIQUE ANCHOR to change")
        else:
            lines.append(f"line {i}: lorem ipsum dolor sit amet consectetur")
    return "\n".join(lines) + "\n"


class _CapturingMCP:
    """A stand-in for FastMCP that just captures the functions a register_* call
    decorates, so the real tool wrapper can be driven without the framework."""

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[kwargs.get("name", fn.__name__)] = fn
            return fn

        return decorator


def _wrapper_fn():
    """Register the real edit_file tool onto a capturing mcp and hand back the fn."""
    mcp = _CapturingMCP()
    # service is unused by the validation paths we test on the wrapper, but the
    # wrapper needs one to forward to; a throwaway rooted at cwd is fine.
    register_edit_file(mcp, FileSystemService(root=Path(tempfile.gettempdir())))
    return mcp.tools["edit_file"]


def _raises(fn, exc) -> Exception | None:
    try:
        fn()
    except exc as caught:
        return caught
    return None


# ---------------------------------------------------------------------------
# The service: the plan's verify and the fail-safe neighbours
# ---------------------------------------------------------------------------
def test_surgical_edit_changes_only_the_target_region() -> None:
    # The plan's verify, half one: a large file, one region changed, the rest
    # left byte-for-byte. Proven two ways - the whole file equals the original
    # with just that span swapped, and a line-by-line diff finds exactly one
    # changed line.
    with tempfile.TemporaryDirectory() as root:
        original = _big_file_text(target_line=250)
        path = _write(root, "big.py", original)
        anchor = "line 250: THE ONE UNIQUE ANCHOR to change"
        replacement = "line 250: the anchor has been surgically replaced"

        result = _service(root).edit_file(path, anchor, replacement)

        expected = original.replace(anchor, replacement, 1)
        assert _read(path) == expected                       # whole file, byte-for-byte
        before, after = original.splitlines(), _read(path).splitlines()
        differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert differing == [250]                            # only the target line moved
        assert result["replacements"] == 1


def test_non_matching_anchor_fails_safely_without_editing() -> None:
    # The plan's verify, half two: an anchor that isn't there fails, and the file
    # is untouched - it never edits the wrong place as a consolation.
    with tempfile.TemporaryDirectory() as root:
        original = _big_file_text()
        path = _write(root, "big.py", original)

        err = _raises(
            lambda: _service(root).edit_file(path, "text that is nowhere in the file", "x"),
            ValueError,
        )
        assert err is not None and "not found" in str(err).lower()
        assert _read(path) == original                       # left exactly as it was


def test_ambiguous_anchor_is_refused_unless_replace_all() -> None:
    # An anchor that matches more than once is the classic "edit the wrong place"
    # trap. Without replace_all it refuses and changes nothing; the message says
    # how to proceed (add context, or opt into replace_all).
    with tempfile.TemporaryDirectory() as root:
        original = "x = 1\nx = 1\nx = 1\n"                    # 'x = 1' three times
        path = _write(root, "dup.py", original)

        err = _raises(lambda: _service(root).edit_file(path, "x = 1", "x = 2"), ValueError)
        assert err is not None
        message = str(err).lower()
        assert "3 places" in message and "replace_all" in message
        assert _read(path) == original                       # unchanged


def test_replace_all_changes_every_occurrence() -> None:
    with tempfile.TemporaryDirectory() as root:
        path = _write(root, "dup.py", "x = 1\nx = 1\nx = 1\n")

        result = _service(root).edit_file(path, "x = 1", "x = 2", replace_all=True)

        assert _read(path) == "x = 2\nx = 2\nx = 2\n"
        assert result["replacements"] == 3


def test_empty_anchor_is_rejected() -> None:
    # An empty anchor "matches" between every character; replacing it would
    # corrupt the file, so it's refused before any ambiguity logic runs.
    with tempfile.TemporaryDirectory() as root:
        original = "keep me intact\n"
        path = _write(root, "f.txt", original)

        err = _raises(lambda: _service(root).edit_file(path, "", "junk"), ValueError)
        assert err is not None and "non-empty" in str(err).lower()
        assert _read(path) == original


def test_identical_old_and_new_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as root:
        original = "same\n"
        path = _write(root, "f.txt", original)

        err = _raises(lambda: _service(root).edit_file(path, "same", "same"), ValueError)
        assert err is not None and "identical" in str(err).lower()
        assert _read(path) == original


def test_new_string_can_delete_the_anchor() -> None:
    # An empty new_string is allowed: it deletes the anchor span. (Empty *old*
    # is what's forbidden, not empty *new*.)
    with tempfile.TemporaryDirectory() as root:
        path = _write(root, "f.txt", "keep <REMOVE ME> keep\n")

        result = _service(root).edit_file(path, "<REMOVE ME> ", "")

        assert _read(path) == "keep keep\n"
        assert result["replacements"] == 1


def test_multiline_anchor_is_replaced_surgically() -> None:
    with tempfile.TemporaryDirectory() as root:
        original = "before\ndef old():\n    return 1\nafter\n"
        path = _write(root, "m.py", original)
        anchor = "def old():\n    return 1"
        replacement = "def new():\n    return 2"

        _service(root).edit_file(path, anchor, replacement)

        assert _read(path) == "before\ndef new():\n    return 2\nafter\n"


def test_edit_missing_file_raises() -> None:
    with tempfile.TemporaryDirectory() as root:
        err = _raises(
            lambda: _service(root).edit_file(os.path.join(root, "nope.txt"), "a", "b"),
            FileNotFoundError,
        )
        assert err is not None


def test_edit_stays_inside_the_workspace_root() -> None:
    # edit_file inherits the service's path fence: a path escaping the root is
    # rejected and the outside file is never touched.
    with tempfile.TemporaryDirectory() as parent:
        root = os.path.join(parent, "workspace")
        os.makedirs(root)
        outside = _write(parent, "secret.txt", "do not touch")

        err = _raises(
            lambda: _service(root).edit_file("../secret.txt", "do not touch", "hacked"),
            PermissionError,
        )
        assert err is not None
        assert _read(outside) == "do not touch"              # the fence held


def test_return_payload_reports_replacements_and_byte_sizes() -> None:
    with tempfile.TemporaryDirectory() as root:
        path = _write(root, "f.txt", "aaa BBB ccc")
        result = _service(root).edit_file(path, "BBB", "BBBB")
        assert result["path"] == str(Path(path).resolve())
        assert result["replacements"] == 1
        assert result["bytes_before"] == len(b"aaa BBB ccc")
        assert result["bytes_after"] == len(b"aaa BBBB ccc")
        assert result["encoding"] == "utf-8"


# ---------------------------------------------------------------------------
# The tool wrapper: its input validation forwards to the service
# ---------------------------------------------------------------------------
def test_wrapper_forwards_a_valid_edit_to_the_service() -> None:
    # The real wrapper, driven without FastMCP, does the edit through the service.
    with tempfile.TemporaryDirectory() as root:
        path = _write(root, "f.txt", "hello world")
        mcp = _CapturingMCP()
        register_edit_file(mcp, FileSystemService(root=Path(root).resolve()))
        edit = mcp.tools["edit_file"]

        result = edit(path=path, old_string="world", new_string="there")

        assert _read(path) == "hello there"
        assert result["replacements"] == 1


def test_wrapper_rejects_empty_path_and_anchor() -> None:
    edit = _wrapper_fn()
    assert _raises(lambda: edit(path="  ", old_string="a", new_string="b"), ValueError) is not None
    assert _raises(lambda: edit(path="f.txt", old_string="", new_string="b"), ValueError) is not None


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_surgical_edit_changes_only_the_target_region,
        test_non_matching_anchor_fails_safely_without_editing,
        test_ambiguous_anchor_is_refused_unless_replace_all,
        test_replace_all_changes_every_occurrence,
        test_empty_anchor_is_rejected,
        test_identical_old_and_new_is_rejected,
        test_new_string_can_delete_the_anchor,
        test_multiline_anchor_is_replaced_surgically,
        test_edit_missing_file_raises,
        test_edit_stays_inside_the_workspace_root,
        test_return_payload_reports_replacements_and_byte_sizes,
        test_wrapper_forwards_a_valid_edit_to_the_service,
        test_wrapper_rejects_empty_path_and_anchor,
    ]
    failures: list = []
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - edit_file:")
        for line in failures:
            print("  -", line)
        return 1
    print(
        f"PASS - edit_file: {len(tests)} tests "
        "(service: surgical + fail-safe x11, wrapper: forwarding + validation x2)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

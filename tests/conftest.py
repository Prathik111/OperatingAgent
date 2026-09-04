"""Shared pytest configuration and fixtures for the OperatingAgent suite.

The suite is a pyramid: many ``tests/unit`` tests, fewer ``tests/integration``,
a handful of ``tests/e2e``. Level markers are applied automatically from the
directory a test lives in (see ``pytest_collection_modifyitems``), so ``-m
unit`` / ``-m integration`` / ``-m e2e`` select a tier without every module
declaring a ``pytestmark``. ``regression`` is hand-applied to the tests that
guard a fixed bug or a safety invariant.

Three jobs, in order:

1. **Environment bootstrap (at import time).** Several modules read the
   environment when they are *imported*, not when they are called —
   ``file_server.server`` instantiates ``FileSystemService()`` at module scope
   (pinning ``FILE_SERVER_ROOT``), and ``TerminalService`` snapshots
   ``TERMINAL_SERVER_ALLOWED_COMMANDS``. Anything that adjusts those variables
   from inside a fixture is already too late, so it happens here, before pytest
   collects (and therefore imports) a single test module.

2. **Hermetic defaults.** Unit, integration and e2e tests must never reach
   Langfuse, an LLM provider, or a live MCP gateway. Credentials are cleared for
   the whole session so a developer's populated ``.env`` cannot turn a unit test
   into a network call. The cleared values are stashed in
   ``CLEARED_CREDENTIALS`` so the opt-in ``tests/*/live`` tiers can restore them
   per-test via ``monkeypatch`` — see ``tests/support/live.py``.

3. **Level marking.** Applied by path, so moving a test between tiers is a
   ``git mv`` and nothing else.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = Path(__file__).resolve().parent

# ===========================================================================
# 1. Environment bootstrap — runs on import, before any test module is loaded
# ===========================================================================

#: Credentials that must be absent so tracing/providers stay disabled.
_CLEARED_VARS = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_RELEASE",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
)

#: Whatever ``_CLEARED_VARS`` held before being cleared. The live tiers read
#: this (never ``os.environ``) so shell-exported credentials work as well as
#: ``.env.test`` ones, while the default run stays hermetic.
CLEARED_CREDENTIALS: dict[str, str] = {}

#: Values the suite pins regardless of what the developer's .env says.
_FORCED_VARS = {
    "LANGFUSE_HOST": "http://localhost:3000",
    "LANGFUSE_TRACING_ENVIRONMENT": "test",
    "TERMINAL_SERVER_ALLOWED_COMMANDS": "cat,dir,echo,ls,pwd,type,where,which",
}


def _load_env_test() -> None:
    """Load ``.env.test`` if present. Optional — the defaults below suffice."""
    env_test = REPO_ROOT / ".env.test"
    if not env_test.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:  # python-dotenv is a dev dep; tolerate absence
        return
    # override=True: the test file wins over an already-exported shell value,
    # which is the whole point of having a test-specific environment.
    load_dotenv(env_test, override=True)


def _bootstrap_environment() -> Path:
    """Pin the environment for the session and return the file-server root.

    The workspace root is a real temporary directory rather than a ``tmp_path``
    because it must exist before ``file_server.server`` is imported, and
    ``tmp_path`` is only available once a test is running.
    """
    _load_env_test()

    for name in _CLEARED_VARS:
        value = os.environ.pop(name, None)
        if value:
            CLEARED_CREDENTIALS[name] = value
    for name, value in _FORCED_VARS.items():
        os.environ[name] = value

    configured = os.environ.get("OPERATING_AGENT_TEST_WORKSPACE")
    workspace = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(tempfile.mkdtemp(prefix="operating-agent-tests-")).resolve()
    )
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["FILE_SERVER_ROOT"] = str(workspace)
    return workspace


#: Workspace root pinned for the whole session; see ``file_server_root``.
SESSION_WORKSPACE: Path = _bootstrap_environment()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Remove the session workspace, unless the developer supplied their own."""
    if os.environ.get("OPERATING_AGENT_TEST_WORKSPACE"):
        return
    shutil.rmtree(SESSION_WORKSPACE, ignore_errors=True)


# ===========================================================================
# 2. Level marking by directory
# ===========================================================================

#: Top-level directory under ``tests/`` -> the pyramid tier it represents.
_TIER_MARKERS = {"unit": "unit", "integration": "integration", "e2e": "e2e"}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark each test with its tier, and anything under a ``live/`` directory as
    ``live`` + ``slow``.

    Doing this by path means a test's tier is wherever it sits on disk — there is
    no second source of truth to forget to update, and ``-m unit`` can never
    silently miss a module that omitted a ``pytestmark``.
    """
    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            parts = path.relative_to(TESTS_ROOT).parts
        except ValueError:  # a test outside tests/ — nothing to infer
            continue

        tier = _TIER_MARKERS.get(parts[0]) if parts else None
        if tier:
            item.add_marker(getattr(pytest.mark, tier))
        # parts[:-1] is the directory chain (excludes the module filename).
        if "live" in parts[:-1]:
            item.add_marker(pytest.mark.live)
            item.add_marker(pytest.mark.slow)


# ===========================================================================
# 3. Cross-cutting fixtures
# ===========================================================================


@pytest.fixture(scope="session")
def file_server_root() -> Path:
    """The directory ``FILE_SERVER_ROOT`` points at for this session."""
    return SESSION_WORKSPACE


@pytest.fixture
def workspace(file_server_root: Path, request: pytest.FixtureRequest) -> Path:
    """A per-test subdirectory inside the pinned file-server root.

    Tests get isolation from each other while every path still resolves inside
    ``FILE_SERVER_ROOT``, so ``FileSystemService`` accepts it.
    """
    # Node ids contain characters that are illegal in Windows paths.
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in request.node.name)
    directory = file_server_root / safe_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(autouse=True)
def reset_observability_singletons() -> Any:
    """Reset ``observability.client``'s module-level cache around every test.

    ``init_tracing`` is intentionally idempotent via ``_client`` /
    ``_initialised`` globals. Without this, the first test to touch tracing
    would decide the answer for every later one.
    """
    from observability import client as observability_client

    def _clear() -> None:
        observability_client._client = None
        observability_client._initialised = False

    _clear()
    yield
    _clear()


@pytest.fixture
def live_tests_enabled() -> bool:
    """Whether tests marked ``integration`` may use real services."""
    return os.environ.get("OPERATING_AGENT_ENABLE_LIVE_TESTS", "0") == "1"


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a throwaway module in ``sys.modules`` for the duration of a test.

    Used to stand in for optional third-party SDKs that are not installed
    (``langchain_openai``, ``langfuse.langchain``, ...) so the code paths that
    import them can still be exercised.
    """

    def _install(name: str, **attributes: Any) -> Any:
        import types

        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return _install

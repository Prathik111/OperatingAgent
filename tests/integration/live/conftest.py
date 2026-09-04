"""Gating for the live integration tier.

Importing ``live_env`` registers it as an autouse fixture for this directory, so
every test here restores real credentials for its own duration and skips when
``OPERATING_AGENT_ENABLE_LIVE_TESTS`` is not 1. See ``tests/support/live.py``.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from tests.support.live import live_env  # noqa: F401  (registers the autouse fixture)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use psycopg-compatible event loops for async live checks on Windows."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()

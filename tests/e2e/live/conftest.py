"""Gating for the live e2e tier.

Importing ``live_env`` registers it as an autouse fixture for this directory, so
every test here restores real credentials for its own duration and skips when
``OPERATING_AGENT_ENABLE_LIVE_TESTS`` is not 1. See ``tests/support/live.py``.
"""

from __future__ import annotations

from tests.support.live import live_env  # noqa: F401  (registers the autouse fixture)

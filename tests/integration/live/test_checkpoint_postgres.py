"""Live integration: the Postgres checkpointer against a real database.

The checkpointer is what makes the executor->verifier loop resumable. Its policy
(disable switch, memory aliases, loud failure on misconfiguration) is pinned
hermetically in ``tests/unit/agent_langgraph/test_checkpoint_factory.py``; this
module is the one case that needs a real server.

Requires ``OPERATING_AGENT_TEST_POSTGRES_URL`` in addition to the live flag, and
skips without it — most developers will not have a database to hand.
"""

from __future__ import annotations

import os

import pytest
from agent_langgraph.checkpoint_factory import CheckpointFactory

from tests.support.langgraph import build_agent_config


def _url() -> str:
    url = os.environ.get("OPERATING_AGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set OPERATING_AGENT_TEST_POSTGRES_URL to run the Postgres tests")
    return url


async def test_postgres_saver_is_built_against_a_real_database() -> None:
    pytest.importorskip("langgraph.checkpoint.postgres")
    factory = CheckpointFactory(
        build_agent_config(checkpoint_backend="postgres", connection_string=_url())
    )
    async with factory.open_checkpointer() as saver:
        assert saver is not None
        assert type(saver).__name__ == "AsyncPostgresSaver"

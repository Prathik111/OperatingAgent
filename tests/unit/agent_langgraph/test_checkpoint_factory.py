"""Tests for ``agent_langgraph.checkpoint_factory.CheckpointFactory``.

The checkpointer is what makes the executor→verifier loop resumable. The
policy pinned here: honour the disable switch, treat the memory aliases
uniformly, and fail loudly (never silently fall back to memory) when a
durable backend is misconfigured or its driver is missing.

The one case that needs a real database lives in
``tests/integration/live/test_checkpoint_postgres.py``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from agent_langgraph.checkpoint_factory import CheckpointFactory
from agent_langgraph.graph.state import AgentPlan, Finding, PlanStep
from common.enums import RunStatus, TaskStatus, VerificationResult, WorkflowPhase
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from tests.support.langgraph import build_agent_config


def factory(**config_kwargs) -> CheckpointFactory:
    return CheckpointFactory(build_agent_config(**config_kwargs))


async def test_disabled_checkpoints_returns_none() -> None:
    async with factory(enable_checkpoints=False).open_checkpointer() as saver:
        assert saver is None


@pytest.mark.parametrize("backend", ["memory", "inmemory", "in_memory", "MEMORY"])
async def test_memory_backends_return_memory_saver(backend: str) -> None:
    async with factory(checkpoint_backend=backend).open_checkpointer() as saver:
        assert isinstance(saver, MemorySaver)


@pytest.mark.regression
async def test_memory_serializer_round_trips_checkpointed_application_types() -> None:
    async with factory(checkpoint_backend="memory").open_checkpointer() as saver:
        assert isinstance(saver, MemorySaver)
        state = {
            "plan": AgentPlan(
                summary="test",
                reasoning="exercise checkpoint serialization",
                steps=[
                    PlanStep(
                        id=1,
                        description="verify",
                        verification=VerificationResult.VERIFIED,
                        status=RunStatus.COMPLETED,
                    )
                ],
            ),
            "findings": [
                Finding(
                    step_id=1,
                    description="checked",
                    detail="working",
                    phase=WorkflowPhase.INVESTIGATE,
                )
            ],
            "workflow_phase": WorkflowPhase.REMEDIATE,
            "status": TaskStatus.EXECUTING,
        }

        restored = saver.serde.loads_typed(saver.serde.dumps_typed(state))

    assert isinstance(restored["plan"], AgentPlan)
    assert isinstance(restored["plan"].steps[0], PlanStep)
    assert restored["plan"].steps[0].verification is VerificationResult.VERIFIED
    assert restored["plan"].steps[0].status is RunStatus.COMPLETED
    assert isinstance(restored["findings"][0], Finding)
    assert restored["findings"][0].phase is WorkflowPhase.INVESTIGATE
    assert restored["workflow_phase"] is WorkflowPhase.REMEDIATE
    assert restored["status"] is TaskStatus.EXECUTING


@pytest.mark.regression
async def test_sqlite_saver_is_setup_and_closed(tmp_path) -> None:
    path = tmp_path / "checkpoints.db"
    async with factory(
        checkpoint_backend="sqlite", connection_string=str(path)
    ).open_checkpointer() as saver:
        assert saver is not None
        assert path.exists()


@pytest.mark.regression
async def test_unknown_backend_raises_value_error() -> None:
    """Never silently fall back to memory — that would lose durability in
    production without anyone noticing."""
    with pytest.raises(ValueError, match="unknown checkpoint backend"):
        async with factory(checkpoint_backend="cassandra").open_checkpointer():
            pass


@pytest.mark.regression
async def test_postgres_without_connection_string_raises() -> None:
    with pytest.raises(ValueError, match="connection_string is not set"):
        async with factory(
            checkpoint_backend="postgres", connection_string=None
        ).open_checkpointer():
            pass


@pytest.mark.regression
async def test_postgres_without_driver_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the postgres driver is missing the factory must raise a
    ModuleNotFoundError that names the package to install — never fall back to
    memory, which would silently lose durability in production.

    The driver ships as a transitive dependency in some environments and not
    others, so rather than rely on its absence we force *its* import to fail.
    ``__import__`` is patched (not ``sys.modules[...] = None``, which raises a
    bare ``ImportError`` the factory's ``except ModuleNotFoundError`` would let
    through) so a genuine ``ModuleNotFoundError`` reaches the code under test.
    """
    import builtins

    real_import = builtins.__import__

    def _fail_postgres_import(name: str, *args, **kwargs):
        if name == "langgraph.checkpoint.postgres.aio":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_postgres_import)

    fac = factory(
        checkpoint_backend="postgres",
        connection_string="postgresql://localhost/db",
    )
    with pytest.raises(ModuleNotFoundError, match="langgraph-checkpoint-postgres"):
        async with fac.open_checkpointer():
            pass


async def test_postgres_saver_is_setup_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    calls: list[str] = []

    class FakeSaver:
        async def setup(self) -> None:
            calls.append("setup")

    @asynccontextmanager
    async def fake_from_conn_string(connection_string: str, *, serde):
        assert connection_string == "postgresql://localhost/db"
        assert isinstance(serde, JsonPlusSerializer)
        calls.append("enter")
        try:
            yield FakeSaver()
        finally:
            calls.append("exit")

    monkeypatch.setattr(
        AsyncPostgresSaver,
        "from_conn_string",
        staticmethod(fake_from_conn_string),
    )

    fac = factory(
        checkpoint_backend="postgres",
        connection_string="postgresql://localhost/db",
    )
    async with fac.open_checkpointer() as saver:
        assert isinstance(saver, FakeSaver)
        assert calls == ["enter", "setup"]

    assert calls == ["enter", "setup", "exit"]

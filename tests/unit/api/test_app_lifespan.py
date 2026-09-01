from __future__ import annotations

import api.app as app_module
import pytest
from api.config import ApiSettings
from api.repository.memory import InMemoryTaskRepository


class _Pool:
    def __init__(self) -> None:
        self.wait: bool | None = None
        self.closed = False

    async def open(self, *, wait: bool = False) -> None:
        self.wait = wait

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_postgres_pool_is_ready_before_startup_completes(monkeypatch) -> None:
    pool = _Pool()
    repository = InMemoryTaskRepository()
    monkeypatch.setattr(
        app_module,
        "build_repository",
        lambda settings: (repository, pool),
    )
    monkeypatch.setattr(app_module, "build_orchestrators", lambda *args, **kwargs: {})
    monkeypatch.setattr(app_module, "init_tracing", lambda: None)
    monkeypatch.setattr(app_module, "flush", lambda: None)
    monkeypatch.setattr(app_module, "shutdown", lambda: None)

    app = app_module.create_app(
        ApiSettings(repository_backend="postgres", database_url="postgresql://test")
    )

    async with app.router.lifespan_context(app):
        assert pool.wait is True

    assert pool.closed is True

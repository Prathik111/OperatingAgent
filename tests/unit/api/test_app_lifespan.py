from __future__ import annotations

import api.app as app_module
import httpx
import pytest
from api.config import ApiSettings
from api.repository.memory import InMemoryTaskRepository
from common.enums import AgentTrack


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


@pytest.mark.asyncio
async def test_real_lifespan_exposes_common_and_native_health_and_task_routes() -> None:
    app = app_module.create_app(
        ApiSettings(
            repository_backend="memory",
            default_track=AgentTrack.NATIVE,
            llm_provider="ollama",
            llm_model="llama3.1",
        )
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            health = await client.get("/health")
            native_health = await client.get("/native/health")
            created = await client.post(
                "/tasks",
                json={
                    "goal": "smoke",
                    "track": "native",
                    "thread_id": "smoke-thread",
                },
            )
            await app.state.task_service.wait_idle()
            status = await client.get(f"/tasks/{created.json()['id']}")

    assert health.status_code == 200
    assert set(health.json()["tracks"]) == {"native", "langgraph"}
    assert native_health.status_code == 200
    assert native_health.json()["agents"] == ["build"]
    assert "llama3.1" in native_health.json()["models"]
    assert created.status_code == 202
    assert status.status_code == 200
    # The shared native track now invokes the real AgentService. This hermetic
    # test has no Ollama process, so the loop reports a clean failed run instead
    # of the former native echo stub's unconditional completion.
    assert status.json()["status"] == "failed"

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
async def test_postgres_failure_without_fallback_fails_startup(monkeypatch) -> None:
    """Regression (P0-7): with the default fallback, a dead Postgres store
    fails startup instead of silently running on memory."""

    def boom(settings):
        raise RuntimeError("pg down")

    monkeypatch.setattr(app_module, "build_repository", boom)
    monkeypatch.setattr(app_module, "build_orchestrators", lambda *args, **kwargs: {})
    monkeypatch.setattr(app_module, "init_tracing", lambda: None)
    monkeypatch.setattr(app_module, "flush", lambda: None)
    monkeypatch.setattr(app_module, "shutdown", lambda: None)

    app = app_module.create_app(
        ApiSettings(repository_backend="postgres", database_url="postgresql://test")
    )
    with pytest.raises(RuntimeError, match="pg down"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover - startup must not get here


@pytest.mark.asyncio
async def test_postgres_failure_with_explicit_fallback_serves_degraded(monkeypatch) -> None:
    """Regression (P0-7): an explicitly configured fallback boots, but
    /health reports degraded with the reason instead of a quiet all-clear."""

    def boom(settings):
        raise RuntimeError("pg down")

    monkeypatch.setattr(app_module, "build_repository", boom)
    monkeypatch.setattr(app_module, "build_orchestrators", lambda *args, **kwargs: {})
    monkeypatch.setattr(app_module, "init_tracing", lambda: None)
    monkeypatch.setattr(app_module, "flush", lambda: None)
    monkeypatch.setattr(app_module, "shutdown", lambda: None)

    app = app_module.create_app(
        ApiSettings(
            repository_backend="postgres",
            database_url="postgresql://test",
            repository_fallback="memory",
        )
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            health = await client.get("/health")

    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "degraded"
    assert body["repository"] == "memory"
    # Exactly one incident is recorded: the task store fell back explicitly,
    # which also flips the effective config (backend + DSN) that the native
    # store follows — so the native track lands on memory as a consequence of
    # the recorded decision, not as a second silent downgrade.
    assert len(body["degraded"]) == 1
    assert "task repository" in body["degraded"][0]
    assert "memory fallback" in body["degraded"][0]


@pytest.mark.asyncio
async def test_native_connect_failure_with_explicit_fallback_is_recorded(monkeypatch) -> None:
    """The native store degrades independently of the task store.

    Task repo builds fine while the native postgres connection fails fast:
    with an explicit memory fallback the app boots, and /health records
    exactly the native incident — nothing silent, nothing attributed to the
    wrong store.
    """
    import agent_native.postgres as native_postgres
    from agent_native.database import MemoryDatabase

    class _UnconnectableDatabase(MemoryDatabase):
        async def connect(self) -> None:
            raise RuntimeError("connection refused")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        native_postgres, "PostgresDatabase", lambda dsn: _UnconnectableDatabase()
    )
    monkeypatch.setattr(
        app_module,
        "build_repository",
        lambda settings: (InMemoryTaskRepository(), None),
    )
    monkeypatch.setattr(app_module, "build_orchestrators", lambda *args, **kwargs: {})
    monkeypatch.setattr(app_module, "init_tracing", lambda: None)
    monkeypatch.setattr(app_module, "flush", lambda: None)
    monkeypatch.setattr(app_module, "shutdown", lambda: None)

    app = app_module.create_app(
        ApiSettings(
            repository_backend="postgres",
            database_url="postgresql://test",
            repository_fallback="memory",
        )
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            health = await client.get("/health")

    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "degraded"
    assert len(body["degraded"]) == 1
    assert "native" in body["degraded"][0]
    assert "memory fallback" in body["degraded"][0]


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

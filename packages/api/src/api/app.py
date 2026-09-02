"""Application factory: wiring, lifespan, CORS and error handling.

``create_app`` builds the ``FastAPI`` instance and includes the routers; the
lifespan constructs the runtime collaborators (repository, orchestrators,
broker, approvals, task service) and tears them down cleanly on shutdown —
cancelling in-flight runs, closing every stream, flushing tracing and closing
the connection pool.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from observability import flush, init_tracing, shutdown
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import ApiSettings
from .environment import load_environment
from .errors import register_exception_handlers
from .orchestration.factory import build_orchestrators
from .repository.factory import build_repository
from .routers import approvals, health, stream, tasks, threads
from .security import SecurityHeadersMiddleware
from .services.approval_gateway import ApprovalGateway
from .services.event_broker import EventBroker
from .services.task_service import TaskService

# Native-track imports are optional at import-time so the Task API still boots
# even if agent-native is not installed; the lifespan will degrade gracefully.
try:
    from .native.routers import events as native_events
    from .native.routers import health as native_health
    from .native.routers import messages as native_messages
    from .native.routers import permissions as native_permissions
    from .native.routers import runs as native_runs
    from .native.routers import sessions as native_sessions

    _NATIVE_ROUTERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _NATIVE_ROUTERS_AVAILABLE = False
    native_events = native_health = native_messages = native_permissions = native_runs = native_sessions = None  # type: ignore

log = logging.getLogger(__name__)

def create_app(settings: ApiSettings | None = None) -> FastAPI:
    if settings is None:
        load_environment()
        settings = ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_tracing()  # idempotent; a no-op when Langfuse creds are absent

        repository, pool = build_repository(settings)
        if pool is not None:
            # A background-only open lets startup report ready even when no
            # connection can be established; the first request then hangs until
            # PoolTimeout. Wait for one usable connection at the readiness
            # boundary so a bad DSN fails startup immediately and clearly.
            await pool.open(wait=True)

        approval_gateway = ApprovalGateway(
            threshold=settings.approval_threshold,
            repository=repository,
        )
        orchestrators = build_orchestrators(
            settings, approval_handler=approval_gateway
        )
        broker = EventBroker()
        background: set[asyncio.Task] = set()

        app.state.repository = repository
        app.state.broker = broker
        app.state.approvals = approval_gateway
        app.state.background = background
        app.state.task_service = TaskService(
            orchestrators=orchestrators,
            repository=repository,
            broker=broker,
            approvals=approval_gateway,
            settings=settings,
            background=background,
        )

        # Native-track runtime — parallel, isolated from Task repository
        native_runtime = None
        native_service = None
        native_pool = None
        native_background: set[asyncio.Task] = set()
        native_cancels: dict = {}
        try:
            from .native.runtime import build_native_database, wire_native_models

            try:
                native_db, native_pool = build_native_database(settings)
                # For postgres, open the native pool explicitly
                if native_pool is not None and hasattr(native_pool, "connect"):
                    # PostgresDatabase owns asyncpg pool
                    try:
                        await native_pool.connect()
                    except Exception as exc:
                        # Only fall back to memory in explicit dev mode; otherwise mark unavailable
                        backend = (getattr(settings, "repository_backend", "") or "").lower()
                        if backend in ("memory", "inmemory", "in_memory"):
                            log.warning("Native Postgres connect failed, falling back to memory (dev): %s", exc)
                            from agent_native.database import MemoryDatabase

                            native_db, native_pool = MemoryDatabase(), None
                        else:
                            log.warning("Native Postgres connect failed, marking native runtime unavailable: %s", exc)
                            raise
                # Enforce schema only if the DB reports missing migrations
                if native_pool is not None:
                    apply_schema = getattr(native_db, "apply_schema", None)
                    if callable(apply_schema):
                        try:
                            await apply_schema()  # type: ignore[operator]  # pyright: ignore[reportGeneralTypeIssues]
                        except Exception as exc:
                            log.warning("Native schema check failed (continuing): %s", exc)

                from agent_native.service import AgentRuntime, AgentService

                native_runtime = AgentRuntime(database=native_db)
                wire_native_models(native_runtime)
                native_service = AgentService(native_runtime)
                log.info(
                    "Native runtime ready: db=%s agents=%s models=%s",
                    type(native_db).__name__,
                    list(getattr(native_runtime, "agents", {}).keys()),
                    list(getattr(getattr(native_runtime, "models", None), "_models", {}).keys()) if hasattr(getattr(native_runtime, "models", None), "_models") else [],
                )
            except Exception as exc:
                log.warning("Native runtime not available: %s", exc)
                native_runtime = None
                native_service = None
        except Exception as exc:
            log.debug("Native package not importable: %s", exc)

        app.state.native_runtime = native_runtime
        app.state.native_service = native_service
        app.state.native_background = native_background
        app.state.native_cancels = native_cancels
        app.state.native_pool = native_pool

        try:
            yield
        finally:
            for run_task in list(background):
                run_task.cancel()
            if background:
                await asyncio.gather(*background, return_exceptions=True)
            await broker.aclose_all()
            await asyncio.gather(
                *(orchestrator.aclose() for orchestrator in orchestrators.values()),
                return_exceptions=True,
            )
            # Native teardown — cancel in-flight sends, close events, MCP, DB
            for t in list(native_background):
                t.cancel()
            if native_background:
                await asyncio.gather(*native_background, return_exceptions=True)
            if native_runtime is not None:
                # Close any MCP providers attached lazily
                for prov in getattr(native_runtime, "_mcp_providers", []) or []:
                    try:
                        await prov.close()
                    except Exception:
                        pass
                # Close native event bus and DB
                try:
                    await native_runtime.events.close()
                except Exception:
                    pass
                try:
                    await native_runtime.database.close()
                except Exception:
                    pass
                # Flush monitoring traces if enabled
                try:
                    for _ in native_runtime.monitoring.shutdown():
                        pass
                except Exception:
                    pass
            elif native_pool is not None:
                try:
                    await native_pool.close()
                except Exception:
                    pass
            flush()
            shutdown()
            if pool is not None:
                await pool.close()

    app = FastAPI(title="OperatingAgent API", version="0.1.0", lifespan=lifespan)
    # Available before the lifespan runs so get_settings works under test.
    app.state.settings = settings

    # allow_credentials with a wildcard origin is rejected by browsers, so only
    # enable credentials when the origins are explicitly enumerated.
    wildcard = "*" in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(settings.allowed_hosts),
    )
    app.add_middleware(SecurityHeadersMiddleware)

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(tasks.router)
    app.include_router(threads.router)
    app.include_router(stream.router)
    app.include_router(approvals.router)
    # Native-track routes — mounted separately so existing paths are untouched
    if _NATIVE_ROUTERS_AVAILABLE:
        assert native_health is not None
        assert native_sessions is not None
        assert native_messages is not None
        assert native_events is not None
        assert native_permissions is not None
        assert native_runs is not None
        app.include_router(native_health.router)
        app.include_router(native_sessions.router)
        app.include_router(native_messages.router)
        app.include_router(native_events.router)
        app.include_router(native_permissions.router)
        app.include_router(native_runs.router)
    return app

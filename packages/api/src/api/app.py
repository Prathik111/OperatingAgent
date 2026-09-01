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
            await pool.open()

        approval_gateway = ApprovalGateway(threshold=settings.approval_threshold)
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
            settings=settings,
            background=background,
        )

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
    return app

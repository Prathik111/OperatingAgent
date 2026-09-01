"""OperatingAgent API package.

The HTTP/WebSocket entry point: submit a goal, stream its run, read the result.
``create_app`` is the ASGI application factory; ``main`` launches it with
uvicorn (the ``api`` console script).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

from .app import create_app

__all__ = ["create_app", "main"]


def _server_loop() -> str | Callable[[], asyncio.AbstractEventLoop]:
    """Return a Uvicorn loop compatible with psycopg's async pool.

    Uvicorn's Windows ``auto`` loop is a ProactorEventLoop, while psycopg's
    asynchronous connections require a selector loop. Passing the factory to
    Uvicorn is deliberate: recent Uvicorn versions create their loop from this
    argument directly, so changing asyncio's global event-loop policy is not
    sufficient.
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop
    return "auto"


def main() -> None:
    """Launch the API with uvicorn using env-sourced settings."""
    import uvicorn

    from .config import ApiSettings
    from .environment import load_environment

    load_environment()
    settings = ApiSettings.from_env()
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        loop=_server_loop(),
    )

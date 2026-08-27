"""OperatingAgent API package.

The HTTP/WebSocket entry point: submit a goal, stream its run, read the result.
``create_app`` is the ASGI application factory; ``main`` launches it with
uvicorn (the ``api`` console script).
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app", "main"]


def main() -> None:
    """Launch the API with uvicorn using env-sourced settings."""
    import uvicorn

    from .config import ApiSettings

    settings = ApiSettings.from_env()
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )

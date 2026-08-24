"""OperatingAgent observability — shared Langfuse tracing.

Public API:
    init_tracing()          initialise the Langfuse client (once, from env)
    get_client()            the Langfuse singleton, or None if disabled
    get_callback_handler()  LangChain/LangGraph CallbackHandler, or None
    flush() / shutdown()    delivery lifecycle for short-lived runs
    LangfuseSettings        env-sourced connection settings
    mask                    the PII/secret masking callable
"""

from .client import (
    flush,
    get_callback_handler,
    get_client,
    init_tracing,
    shutdown,
)
from .masking import mask
from .settings import LangfuseSettings

__all__ = [
    "init_tracing",
    "get_client",
    "get_callback_handler",
    "flush",
    "shutdown",
    "LangfuseSettings",
    "mask",
]


def main() -> None:
    """Print the resolved tracing status — a quick connectivity check."""
    settings = LangfuseSettings.from_env()
    if settings.enabled:
        print(f"Langfuse tracing ENABLED (env={settings.environment}, host={settings.host})")
    else:
        print("Langfuse tracing DISABLED — set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")

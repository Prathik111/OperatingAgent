"""Langfuse connection settings, sourced from the environment.

Credentials are read from environment variables only (never hard-coded or
passed through chat), per Langfuse best practice. Load your ``.env`` before
importing/constructing anything here so the SDK initialises with the right
keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LangfuseSettings:
    """Resolved Langfuse connection settings.

    ``enabled`` is only true when both keys are present — this is what lets the
    whole app run tracing-off in dev/CI without any code changes.
    """

    public_key: str | None
    secret_key: str | None
    host: str
    environment: str
    release: str | None

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)

    @classmethod
    def from_env(cls) -> "LangfuseSettings":
        # Support both LANGFUSE_HOST (SDK-native) and LANGFUSE_BASE_URL (CLI).
        host = (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com"
        )
        return cls(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=host,
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
            release=os.getenv("LANGFUSE_RELEASE"),
        )

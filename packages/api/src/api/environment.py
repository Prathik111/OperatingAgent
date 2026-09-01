"""Load local environment values for API process entrypoints."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_environment() -> None:
    """Load the nearest ``.env`` without replacing exported variables."""
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)

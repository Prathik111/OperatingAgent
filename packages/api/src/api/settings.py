"""Shared API models and helpers for runtime settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


def apply_langfuse_settings(mode: str | None, host: str | None, public_key: str | None, secret_key: str | None) -> dict[str, object]:
    """Apply desktop Langfuse settings to this API process and reload tracing."""
    import os
    from observability import LangfuseSettings, reload_tracing

    selected = (mode or "").strip().lower()
    if selected not in {"", "disabled", "cloud", "local"}:
        raise ValueError("langfuse_mode must be disabled, cloud, or local")
    if selected == "disabled":
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
    else:
        if host is not None and host.strip():
            os.environ["LANGFUSE_HOST"] = host.strip().rstrip("/")
        if public_key is not None:
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key.strip()
        if secret_key is not None:
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key.strip()
    settings = LangfuseSettings.from_env()
    client = reload_tracing(settings)
    return {
        "langfuse_mode": "disabled" if client is None else (selected or "cloud"),
        "langfuse_host": settings.host,
        "langfuse_enabled": client is not None,
        "langfuse_public_key_set": bool(settings.public_key),
        "langfuse_secret_key_set": bool(settings.secret_key),
    }


def current_langfuse_settings() -> dict[str, object]:
    """Return non-secret Langfuse settings for settings forms and health views."""
    from observability import LangfuseSettings

    settings = LangfuseSettings.from_env()
    host = settings.host.rstrip("/")
    local = host.startswith("http://localhost") or host.startswith("http://127.0.0.1") or host.startswith("http://[::1]")
    return {
        "langfuse_mode": "disabled" if not settings.enabled else ("local" if local else "cloud"),
        "langfuse_host": settings.host,
        "langfuse_enabled": settings.enabled,
        "langfuse_public_key_set": bool(settings.public_key),
        "langfuse_secret_key_set": bool(settings.secret_key),
    }


class RuntimeLLMSettings(BaseModel):
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = None
    # Write-only from the desktop settings UI. GET endpoints expose only
    # whether a key is configured, never the secret itself.
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    auto_approve_all: bool | None = Field(
        default=None,
        description="Skip approval prompts (denials still enforced)",
    )
    langfuse_mode: str | None = None
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


_DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen3.5:0.8b",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}

_KNOWN_MODELS: dict[str, list[str]] = {
    "groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
}

_OLLAMA_HOSTS = ("http://localhost:11434", "http://127.0.0.1:11434")


def normalize_provider(value: object) -> str:
    return str(value or "ollama").strip().lower() or "ollama"


def normalize_model(value: object) -> str:
    return str(value or "").strip()


def normalize_base_url(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def normalize_ollama_host(value: object) -> str:
    """Return the Ollama server root (never the ``/api`` path)."""
    text = str(value or "").strip().rstrip("/") or "http://localhost:11434"
    if text.endswith("/api"):
        text = text[: -len("/api")].rstrip("/") or "http://localhost:11434"
    return text


def clean_base_url(provider: str, base_url: str | None) -> str | None:
    """Drop an Ollama URL that was carried over to a cloud provider.

    This is the main cause of Groq's confusing ``404 page not found``: the
    Groq client was pointed at the local Ollama server.
    """
    if base_url and provider != "ollama":
        for host in _OLLAMA_HOSTS:
            if base_url.startswith(host):
                return None
    return base_url


async def ollama_models(base_url: str | None) -> list[str]:
    import httpx

    host = normalize_ollama_host(base_url)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{host}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return []
    models: list[str] = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        name = item.get("name") if isinstance(item, dict) else None
        if name and str(name) not in models:
            models.append(str(name))
    return models


async def provider_models(provider: str, base_url: str | None) -> list[str]:
    provider = normalize_provider(provider)
    if provider == "ollama":
        return await ollama_models(base_url)
    return list(_KNOWN_MODELS.get(provider, []))


def default_model(provider: str, downloaded: list[str] | None = None) -> str:
    provider = normalize_provider(provider)
    if provider == "ollama" and downloaded:
        return downloaded[0]
    return _DEFAULT_MODELS.get(provider, "")


def resolve_model(provider: str, model: str, downloaded: list[str] | None = None) -> str:
    """Return the explicit model, or the provider default when empty."""
    model = normalize_model(model)
    if model:
        return model
    if provider == "ollama" and downloaded:
        return downloaded[0]
    return _DEFAULT_MODELS.get(provider, "")

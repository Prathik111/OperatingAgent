"""Live settings for the native AgentRuntime."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..settings import (
    RuntimeLLMSettings,
    clean_base_url,
    default_model,
    normalize_base_url,
    normalize_model,
    normalize_provider,
    provider_models,
    resolve_model,
)
from .dependencies import get_native_runtime

router = APIRouter(prefix="/native/settings", tags=["native-settings"])
NativeRuntimeDep = Annotated[Any, Depends(get_native_runtime)]

_NATIVE_PROVIDERS = ("groq", "ollama")

#: PATCH fields that reconfigure the model. Anything else (e.g. the allow-all
#: toggle) applies without touching provider validation.
_LLM_PATCH_FIELDS = frozenset(
    {"provider", "model", "base_url", "temperature", "top_p", "max_tokens", "timeout_seconds"}
)


@router.get("")
async def get_native_settings(runtime: NativeRuntimeDep) -> dict[str, Any]:
    agents = list(getattr(runtime, "agents", {}).values())
    config = agents[0] if agents else None
    provider = ""
    if config:
        try:
            provider = runtime.models.get(config.model).provider
        except (KeyError, AttributeError):
            provider = ""
    provider = normalize_provider(provider)
    provider_client = None
    try:
        provider_client = runtime.models.get_provider(config.model) if config else None
    except (KeyError, AttributeError):
        pass
    configured_host = getattr(provider_client, "host", None) or getattr(
        provider_client, "base_url", None
    )
    downloaded = (
        await provider_models(provider, configured_host) if provider == "ollama" else []
    )
    models = list(runtime.models.list_model_names())
    for name in downloaded:
        if name not in models:
            models.append(name)
    auto_approve = bool(getattr(runtime, "auto_approve_all", False))
    return {
        "track": "native",
        "model": getattr(config, "model", "") if config else "",
        "provider": provider,
        "base_url": configured_host,
        "models": models,
        "default_model": default_model(provider, downloaded or None),
        "temperature": getattr(config, "temperature", 0.0),
        "top_p": getattr(config, "top_p", 1.0),
        "max_tokens": getattr(config, "max_output_tokens", None),
        "timeout_seconds": getattr(config, "timeout_seconds", 60),
        "auto_approve_all": bool(auto_approve),
    }


@router.get("/models")
async def list_native_models(
    provider: str = Query(default="ollama"),
    base_url: str | None = Query(default=None),
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    base_url = normalize_base_url(base_url)
    models = await provider_models(provider, base_url)
    return {
        "track": "native",
        "provider": provider,
        "models": models,
        "default_model": default_model(provider, models or None),
    }


@router.patch("")
async def update_native_settings(
    body: RuntimeLLMSettings,
    runtime: NativeRuntimeDep,
) -> dict[str, Any]:
    if body.auto_approve_all is not None and hasattr(runtime, "set_auto_approve_all"):
        try:
            runtime.set_auto_approve_all(bool(body.auto_approve_all))
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not (set(body.model_fields_set) & _LLM_PATCH_FIELDS):
        # Flags-only patch (e.g. just the allow-all toggle): nothing about the
        # model changes, so provider validation must not block it.
        return {
            "track": "native",
            "auto_approve_all": bool(getattr(runtime, "auto_approve_all", False)),
            "applies_to": "new runs",
        }
    agents = list(getattr(runtime, "agents", {}).values())
    current = agents[0] if agents else None
    current_provider = ""
    current_client = None
    if current:
        try:
            current_provider = runtime.models.get(current.model).provider
            current_client = runtime.models.get_provider(current.model)
        except (KeyError, AttributeError):
            pass
    provider = normalize_provider(body.provider or current_provider)
    if provider not in _NATIVE_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported native provider {provider!r}; expected one of {', '.join(_NATIVE_PROVIDERS)}",
        )
    old_base_url = getattr(current_client, "host", None) or getattr(
        current_client, "base_url", None
    )
    base_url = clean_base_url(
        provider,
        normalize_base_url(body.base_url)
        if "base_url" in body.model_fields_set
        else old_base_url,
    )
    downloaded = await provider_models(provider, base_url) if provider == "ollama" else []
    model = resolve_model(
        provider,
        normalize_model(body.model) if body.model is not None else getattr(current, "model", ""),
        downloaded or None,
    )
    if not model:
        raise HTTPException(status_code=422, detail=f"no default model for provider {provider!r}; set model explicitly")
    if provider == "ollama":
        if not downloaded:
            raise HTTPException(
                status_code=422,
                detail=f"Ollama is not reachable at {base_url or 'http://localhost:11434'} or has no models installed",
            )
        if model not in downloaded:
            raise HTTPException(
                status_code=422,
                detail=f"Ollama model {model!r} is not installed; available: {downloaded}",
            )
    try:
        models = runtime.reconfigure_models(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=(
                body.temperature
                if body.temperature is not None
                else getattr(current, "temperature", 0.0)
            ),
            top_p=(
                body.top_p
                if body.top_p is not None
                else getattr(current, "top_p", 1.0)
            ),
            max_tokens=(
                body.max_tokens
                if "max_tokens" in body.model_fields_set
                else getattr(current, "max_output_tokens", None)
            ),
            timeout_seconds=(
                body.timeout_seconds
                if body.timeout_seconds is not None
                else getattr(current, "timeout_seconds", 60)
            ),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for name in downloaded:
        if name not in models:
            models.append(name)
    effective = runtime.config_for("build")
    return {
        "track": "native",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "models": models,
        "default_model": default_model(provider, downloaded or None),
        "temperature": effective.temperature,
        "top_p": effective.top_p,
        "max_tokens": effective.max_output_tokens,
        "timeout_seconds": effective.timeout_seconds,
        "auto_approve_all": bool(getattr(runtime, "auto_approve_all", False)),
        "applies_to": "new runs",
    }

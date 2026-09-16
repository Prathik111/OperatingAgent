"""Settings API for the LLM judge provider.

Mirrors the per-track settings pattern (``langgraph_settings``) but the judge
has no *track*: it is shared across every comparison run.  The Settings
Frontend can PATCH provider/model/api_key/base_url; the Evaluation UI reads
these as sensible defaults and lets the user override provider/model per run
(the key never leaves the backend).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .judge import JUDGE_PROVIDERS, JudgeRuntimeConfig
from .settings import (
    default_model,
    normalize_base_url,
    provider_models,
)

router = APIRouter(prefix="/settings/judge", tags=["judge-settings"])


def _config(request: Request) -> JudgeRuntimeConfig:
    config = getattr(request.app.state, "judge_config", None)
    if config is None:
        raise HTTPException(status_code=503, detail="judge settings not initialized")
    return config


class JudgeSettingsPatch(BaseModel):
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


async def _public(config: JudgeRuntimeConfig) -> dict[str, Any]:
    public = config.public_settings()
    models: list[str] = []
    default_model_name = ""
    try:
        resolved = config.resolve(config.provider, config.model)
        models = await provider_models(resolved.provider, resolved.base_url)
        default_model_name = default_model(resolved.provider, models or None)
    except ValueError:
        # The active provider may have no key configured yet; the Settings
        # page must still render so the user can fill one in.
        models = await provider_models(config.provider, config._resolved_base_url(config.provider))
        default_model_name = default_model(config.provider, models or None)
    return {
        **public,
        "models": models,
        "default_model": default_model_name,
    }


@router.get("")
async def get_judge_settings(request: Request) -> dict[str, Any]:
    config = _config(request)
    return await _public(config)


@router.get("/models")
async def list_judge_models(
    provider: str = Query(default="ollama"),
    base_url: str | None = Query(default=None),
) -> dict[str, Any]:
    provider = (provider or "ollama").strip().lower()
    if provider not in JUDGE_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported provider {provider!r}",
        )
    base_url = normalize_base_url(base_url)
    models = await provider_models(provider, base_url)
    return {
        "provider": provider,
        "models": models,
        "default_model": default_model(provider, models or None),
    }


@router.patch("")
async def update_judge_settings(
    body: JudgeSettingsPatch,
    request: Request,
) -> dict[str, Any]:
    config = _config(request)
    try:
        config.update(
            provider=body.provider,
            model=body.model,
            api_key=body.api_key,
            base_url=body.base_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await _public(config)

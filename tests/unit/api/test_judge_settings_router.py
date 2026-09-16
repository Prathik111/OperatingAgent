"""Judge settings router: provider/model/key roundtrip via the shared backend.

The judge config lives on ``app.state`` (not a per-track service), so this
harness sets it directly instead of overriding a dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from api import create_app
from api.config import ApiSettings
from api.judge import JudgeRuntimeConfig, ProviderCredentials
from httpx import ASGITransport


@pytest.fixture
async def judge_client() -> AsyncIterator[httpx.AsyncClient]:
    config = JudgeRuntimeConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        credentials={
            "groq": ProviderCredentials(api_key="env-placeholder"),
            "openai": ProviderCredentials(),
        },
    )
    app = create_app(ApiSettings(repository_backend="memory"))
    app.state.judge_config = config
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


async def test_judge_settings_get_returns_secret_free_view(judge_client) -> None:
    client = judge_client
    response = await client.get("/settings/judge")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "groq"
    assert payload["api_key_set"] is True
    text = str(payload)
    assert "env-placeholder" not in text
    assert set(payload["providers"]) == {"ollama", "groq", "openai", "anthropic"}
    assert "credentials" in payload
    assert "models" in payload


async def test_judge_settings_patch_writes_active_provider_and_key(judge_client) -> None:
    client = judge_client
    updated = await client.patch(
        "/settings/judge",
        json={"provider": "openai", "model": "gpt-4o-mini", "api_key": "oa-secret"},
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["api_key_set"] is True
    assert "oa-secret" not in str(payload)

    current = await client.get("/settings/judge")
    assert current.json()["provider"] == "openai"


async def test_judge_settings_patch_rejects_unknown_provider(judge_client) -> None:
    client = judge_client
    response = await client.patch("/settings/judge", json={"provider": "together"})
    assert response.status_code == 422


async def test_judge_settings_get_without_key_still_loads() -> None:
    """The Settings page must render even before credentials exist."""
    config = JudgeRuntimeConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        credentials={"groq": ProviderCredentials()},
    )
    app = create_app(ApiSettings(repository_backend="memory"))
    app.state.judge_config = config
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/settings/judge")
        assert response.status_code == 200
        assert response.json()["api_key_set"] is False
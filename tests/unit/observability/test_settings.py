"""Tests for ``observability.settings.LangfuseSettings``.

``enabled`` is the switch the whole tracing stack hinges on — it must be true
only when *both* keys are present, so the app can run tracing-off in dev/CI
with no code changes. The env-var precedence (HOST vs BASE_URL) is also pinned.
"""

from __future__ import annotations

import pytest
from observability.settings import LangfuseSettings


@pytest.fixture
def clean_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every Langfuse variable so each test starts from a blank slate."""
    for name in (
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL", "LANGFUSE_TRACING_ENVIRONMENT", "LANGFUSE_RELEASE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_disabled_when_no_credentials(clean_langfuse_env: None) -> None:
    settings = LangfuseSettings.from_env()
    assert settings.enabled is False
    assert settings.public_key is None
    assert settings.secret_key is None


@pytest.mark.parametrize("present", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
def test_disabled_when_only_one_key_present(
    clean_langfuse_env: None, monkeypatch: pytest.MonkeyPatch, present: str
) -> None:
    monkeypatch.setenv(present, "value")
    assert LangfuseSettings.from_env().enabled is False


def test_enabled_when_both_keys_present(
    clean_langfuse_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-y")
    settings = LangfuseSettings.from_env()
    assert settings.enabled is True
    assert settings.public_key == "pk-lf-x"
    assert settings.secret_key == "sk-lf-y"


def test_default_host_and_environment(clean_langfuse_env: None) -> None:
    settings = LangfuseSettings.from_env()
    assert settings.host == "https://cloud.langfuse.com"
    assert settings.environment == "development"
    assert settings.release is None


def test_host_prefers_langfuse_host_over_base_url(
    clean_langfuse_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://host")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://base")
    assert LangfuseSettings.from_env().host == "https://host"


def test_host_falls_back_to_base_url(
    clean_langfuse_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://base")
    assert LangfuseSettings.from_env().host == "https://base"


def test_environment_and_release_are_read(
    clean_langfuse_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "staging")
    monkeypatch.setenv("LANGFUSE_RELEASE", "abc123")
    settings = LangfuseSettings.from_env()
    assert settings.environment == "staging"
    assert settings.release == "abc123"


def test_settings_are_frozen(clean_langfuse_env: None) -> None:
    settings = LangfuseSettings.from_env()
    with pytest.raises(AttributeError):
        settings.host = "https://elsewhere"  # type: ignore[misc]

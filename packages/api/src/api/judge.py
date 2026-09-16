"""Runtime configuration for the LLM judge's provider.

The judge is deliberately provider-agnostic (:mod:`common.llm_judging`); this
module owns *which* provider/model/credentials a judge run uses. Credentials
are seeded from the process environment at startup (secure deployment-time
bootstrap) and may be changed at runtime through the settings API. Secrets
never leave the backend: responses expose only ``api_key_set`` flags.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from common.config import LLMConfig
from common.llm_judging import LLMJudge

from .settings import (
    clean_base_url,
    default_model,
    normalize_ollama_host,
    normalize_provider,
)

#: Providers the judge can be pointed at. Mirrors the LangGraph track's set.
JUDGE_PROVIDERS: tuple[str, ...] = ("ollama", "groq", "openai", "anthropic")

_DEFAULT_JUDGE_MODELS: dict[str, str] = {
    "ollama": "llama3.1",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}


def _env_api_key(provider: str) -> str:
    return os.getenv(f"{provider.upper()}_API_KEY", "").strip()


def _env_base_url(provider: str) -> str | None:
    return os.getenv(f"{provider.upper()}_BASE_URL", "").strip() or None


@dataclass
class ProviderCredentials:
    """One provider's judge credentials (api_key is never serialized out)."""

    api_key: str = field(default="", repr=False)
    base_url: str | None = None


@dataclass
class JudgeRuntimeConfig:
    """Mutable active judge selection shared by every evaluation."""

    provider: str = ""
    model: str = ""
    credentials: dict[str, ProviderCredentials] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> JudgeRuntimeConfig:
        provider = normalize_provider(
            os.getenv("JUDGE_PROVIDER") or os.getenv("LLM_PROVIDER") or "ollama"
        )
        credentials = {
            name: ProviderCredentials(
                api_key=_env_api_key(name), base_url=_env_base_url(name)
            )
            for name in JUDGE_PROVIDERS
        }
        # A dedicated JUDGE_API_KEY/JUDGE_BASE_URL seeds the active provider
        # without a Settings round-trip.
        active = credentials.setdefault(provider, ProviderCredentials())
        if judge_key := os.getenv("JUDGE_API_KEY", "").strip():
            active.api_key = judge_key
        if judge_url := os.getenv("JUDGE_BASE_URL", "").strip():
            active.base_url = judge_url
        return cls(
            provider=provider,
            model=os.getenv("JUDGE_MODEL", "").strip(),
            credentials=credentials,
        )

    def credentials_for(self, provider: str) -> ProviderCredentials:
        return self.credentials.get(provider) or ProviderCredentials()

    def _resolved_key(self, provider: str) -> str:
        return self.credentials_for(provider).api_key or _env_api_key(provider)

    def _resolved_base_url(self, provider: str) -> str | None:
        configured = self.credentials_for(provider).base_url or _env_base_url(provider)
        return clean_base_url(provider, configured)

    def resolve(self, provider: str = "", model: str = "") -> LLMConfig:
        """Build the ``LLMConfig`` for a judge run.

        ``provider``/``model`` override the active selection (per-run picker);
        empty fields fall back to the Settings default, then the provider's
        built-in model. Missing credentials for a cloud provider are a clear
        validation error, never a silent unauthenticated call.
        """
        chosen_provider = normalize_provider(provider or self.provider or "ollama")
        if chosen_provider not in JUDGE_PROVIDERS:
            raise ValueError(
                f"unsupported judge provider {chosen_provider!r}; "
                f"expected one of {', '.join(JUDGE_PROVIDERS)}"
            )
        chosen_model = (
            (model or self.model or "").strip()
            or _DEFAULT_JUDGE_MODELS.get(chosen_provider)
            or default_model(chosen_provider)
        )
        if not chosen_model:
            raise ValueError(f"no default model for judge provider {chosen_provider!r}")
        base_url = self._resolved_base_url(chosen_provider)
        api_key = self._resolved_key(chosen_provider)
        if chosen_provider == "ollama":
            base_url = normalize_ollama_host(base_url)
        elif not api_key:
            raise ValueError(
                f"An API key is required for the {chosen_provider} judge provider "
                f"(set it in Settings or {chosen_provider.upper()}_API_KEY)"
            )
        return LLMConfig(
            provider=chosen_provider,
            model=chosen_model,
            api_key=api_key,
            base_url=base_url,
        )

    def public_settings(self) -> dict:
        """Safe (secret-free) view for GET responses."""
        active = self.credentials_for(self.provider)
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": active.base_url or _env_base_url(self.provider),
            "api_key_set": bool(self._resolved_key(self.provider))
            if self.provider in JUDGE_PROVIDERS
            else False,
            "providers": list(JUDGE_PROVIDERS),
            "credentials": {
                name: {
                    "api_key_set": bool(self._resolved_key(name)),
                    "base_url": self._resolved_base_url(name),
                }
                for name in JUDGE_PROVIDERS
            },
        }

    def update(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Apply a settings PATCH. ``None`` leaves a field unchanged."""
        if provider is not None and provider.strip():
            normalized = normalize_provider(provider)
            if normalized not in JUDGE_PROVIDERS:
                raise ValueError(
                    f"unsupported judge provider {normalized!r}; "
                    f"expected one of {', '.join(JUDGE_PROVIDERS)}"
                )
            self.provider = normalized
        if model is not None:
            self.model = model.strip()
        active = self.credentials.setdefault(self.provider, ProviderCredentials())
        if api_key is not None:
            # Empty string clears the override and falls back to the env key.
            active.api_key = api_key.strip()
        if base_url is not None:
            active.base_url = base_url.strip() or None


def make_judge_factory(
    config: JudgeRuntimeConfig,
) -> Callable[[str, str], LLMJudge]:
    """Return the ``(provider, model) -> LLMJudge`` factory the service uses.

    The agent-langgraph import is lazy so the API still boots (with the judge
    disabled) when that optional package is unavailable.
    """

    def judge_factory(provider: str, model: str) -> LLMJudge:
        llm = config.resolve(provider, model)
        try:
            from agent_langgraph.runtime.judging import judge_from_llm_config
        except ImportError as exc:
            raise ValueError(
                "the LLM judge provider is unavailable: the agent-langgraph "
                "package is not installed"
            ) from exc
        return judge_from_llm_config(llm)

    return judge_factory

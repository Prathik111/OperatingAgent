"""API settings and the per-track ``AgentConfig`` builder.

``ApiSettings`` is a frozen, env-sourced snapshot (mirroring the ``from_env``
pattern in ``observability.settings``). ``build_agent_config`` assembles an
``AgentConfig`` for a track using only dataclass construction — no model is
built and no network is touched — so it is safe to call from unit tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from common.config import (
    AgentConfig,
    BehaviourConfig,
    CheckpointConfig,
    ExecutionConfig,
    LLMConfig,
    MetadataConfig,
    PromptConfig,
    SandboxConfig,
    ToolPermissionConfig,
    TracingConfig,
)
from common.enums import AgentTrack, RiskLevel
from observability import LangfuseSettings


def _split_origins(raw: str) -> tuple[str, ...]:
    origins = tuple(o.strip() for o in raw.split(",") if o.strip())
    return origins or ("*",)


@dataclass(slots=True, frozen=True)
class ApiSettings:
    """Resolved API configuration for one process."""

    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "info"

    database_url: str | None = field(default=None, repr=False)
    repository_backend: str = "memory"

    cors_origins: tuple[str, ...] = ("*",)

    default_track: AgentTrack = AgentTrack.LANGGRAPH
    approval_threshold: RiskLevel = RiskLevel.REVIEW

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1"
    llm_base_url: str | None = None

    prompt_dir: str = "prompts"
    tracing_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ApiSettings":
        database_url = os.getenv("DATABASE_URL") or None
        backend = os.getenv(
            "API_REPOSITORY_BACKEND", "postgres" if database_url else "memory"
        ).lower()
        return cls(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8080")),
            log_level=os.getenv("API_LOG_LEVEL", "info"),
            database_url=database_url,
            repository_backend=backend,
            cors_origins=_split_origins(os.getenv("API_CORS_ORIGINS", "*")),
            default_track=AgentTrack(os.getenv("API_DEFAULT_TRACK", "langgraph")),
            approval_threshold=RiskLevel(os.getenv("API_APPROVAL_THRESHOLD", "review")),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_model=os.getenv("LLM_MODEL", "llama3.1"),
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            prompt_dir=os.getenv("AGENT_PROMPT_DIR", "prompts"),
            tracing_enabled=LangfuseSettings.from_env().enabled,
        )

    def build_agent_config(self, track: AgentTrack | None = None) -> AgentConfig:
        """Assemble an ``AgentConfig`` for ``track``.

        The api key is read from ``{PROVIDER}_API_KEY`` at call time and defaults
        to an empty string (hermetic). Checkpointing uses Postgres iff a
        ``DATABASE_URL`` is configured, else in-memory. Tracing follows the
        shared Langfuse enablement rule (both keys present).
        """
        provider = self.llm_provider
        api_key = os.getenv(f"{provider.upper()}_API_KEY", "")
        prompts = Path(self.prompt_dir)
        checkpoint_backend = "postgres" if self.database_url else "memory"
        return AgentConfig(
            llm=LLMConfig(
                provider=provider,
                model=self.llm_model,
                api_key=api_key,
                base_url=self.llm_base_url,
            ),
            execution=ExecutionConfig(),
            sandbox=SandboxConfig(),
            permissions=ToolPermissionConfig(),
            checkpoint=CheckpointConfig(
                backend=checkpoint_backend, connection_string=self.database_url
            ),
            tracing=TracingConfig(enabled=self.tracing_enabled),
            behaviour=BehaviourConfig(risk_threshold=self.approval_threshold.value),
            prompts=PromptConfig(
                planner_prompt=prompts / "planner.txt",
                verifier_prompt=prompts / "verifier.txt",
                responder_prompt=prompts / "responder.txt",
            ),
            metadata=MetadataConfig(),
        )

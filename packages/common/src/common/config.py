from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# LLM
# ============================================================

@dataclass(slots=True, frozen=True)
class LLMConfig:
    provider: str
    model: str
    # repr=False keeps the secret out of the generated __repr__ so it never
    # lands in logs, tracebacks, or Langfuse metadata. Field access is
    # unchanged; only the string representation is redacted.
    api_key: str = field(repr=False)

    timeout_seconds: int = 60
    temperature: float = 0.0

    max_tokens: int | None = None

    top_p: float = 1.0

    base_url: str | None = None


# ============================================================
# Execution
# ============================================================

@dataclass(slots=True, frozen=True)
class ExecutionConfig:

    max_iterations: int = 20

    timeout_seconds: int = 300

    retry_attempts: int = 2

    stream: bool = True

    enable_checkpoints: bool = True

    enable_interrupts: bool = True


# ============================================================
# Sandbox
# ============================================================

@dataclass(slots=True, frozen=True)
class SandboxConfig:

    enabled: bool = True

    workspace: Path = Path("./workspace")

    docker_image: str = "python:3.12"

    network_enabled: bool = False

    memory_limit_mb: int = 2048

    cpu_limit: float = 2.0


# ============================================================
# Tool Permissions
# ============================================================

@dataclass(slots=True, frozen=True)
class ToolPermissionConfig:

    file_system: bool = True

    terminal: bool = True

    git: bool = True

    search: bool = True

    knowledge: bool = True

    memory: bool = True


# ============================================================
# Checkpoint
# ============================================================

@dataclass(slots=True, frozen=True)
class CheckpointConfig:

    backend: str = "postgres"

    # A DSN carries the DB password; keep it out of __repr__ (see LLMConfig).
    connection_string: str | None = field(default=None, repr=False)

    namespace: str = "default"


# ============================================================
# Tracing
# ============================================================

@dataclass(slots=True, frozen=True)
class TracingConfig:

    enabled: bool = True

    provider: str = "langfuse"

    project_name: str = "OperatingAgent"

    endpoint: str | None = None


# ============================================================
# Agent Behaviour
# ============================================================

@dataclass(slots=True, frozen=True)
class BehaviourConfig:

    require_verification: bool = True

    require_human_approval: bool = True

    enable_reflection: bool = True

    max_reflections: int = 2

    risk_threshold: str = "review"


# ============================================================
# Prompt
# ============================================================

@dataclass(slots=True, frozen=True)
class PromptConfig:

    planner_prompt: Path

    verifier_prompt: Path

    responder_prompt: Path


# ============================================================
# Metadata
# ============================================================

@dataclass(slots=True, frozen=True)
class MetadataConfig:

    tags: dict[str, str] = field(default_factory=dict)

    custom: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Root Agent Config
# ============================================================

@dataclass(slots=True, frozen=True)
class AgentConfig:

    llm: LLMConfig

    execution: ExecutionConfig

    sandbox: SandboxConfig

    permissions: ToolPermissionConfig

    checkpoint: CheckpointConfig

    tracing: TracingConfig

    behaviour: BehaviourConfig

    prompts: PromptConfig

    metadata: MetadataConfig = field(default_factory=MetadataConfig)
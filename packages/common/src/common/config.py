from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:

    provider: str

    model: str

    temperature: float


@dataclass(slots=True)
class SandboxConfig:

    timeout_seconds: int

    workspace: str


@dataclass(slots=True)
class TracingConfig:

    enabled: bool

    project_name: str
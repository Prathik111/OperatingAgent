"""Runtime configuration for agent-native.

Sources, in precedence order (lowest to highest):
  1. Code defaults defined here.
  2. `AGENT_NATIVE_CONFIG` TOML file (if set / if exists in cwd or package dir).
  3. Environment variables (AGENT_NATIVE_*, plus provider keys).
  4. A `.env` file next to the config file / in cwd (loaded via python-dotenv).

Decision #7: Groq is the primary LLM provider (matches pyproject deps);
Ollama is the optional local fallback. Decision #6: sandbox egress is
deny-all by default; the allowlist lives in the config file and can be
overridden/extended via AGENT_NATIVE_SANDBOX_ALLOWED_HOSTS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv

_PACKAGE_DIR = Path(__file__).resolve().parent

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_APPROVAL_TIMEOUT_S = 120.0
DEFAULT_MAX_CALLS_PER_STEP = 5
DEFAULT_MAX_REPLANS = 3
DEFAULT_TOKEN_BUDGET = 20000
DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"
DEFAULT_SANDBOX_CPU_LIMIT = "1.0"
DEFAULT_SANDBOX_TIME_LIMIT_S = 30
DEFAULT_SANDBOX_MEMORY_LIMIT = "512m"
DEFAULT_DATABASE_URL = "postgresql://agent_native:agent_native@localhost:5432/agent_native"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _find_config_file() -> Path | None:
    explicit = os.environ.get("AGENT_NATIVE_CONFIG")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for candidate in (Path.cwd() / "agent_native.toml", _PACKAGE_DIR / "agent_native.toml"):
        if candidate.is_file():
            return candidate
    return None


@dataclass(slots=True)
class SandboxConfig:
    enabled: bool = True
    image: str = DEFAULT_SANDBOX_IMAGE
    cpu_limit: str = DEFAULT_SANDBOX_CPU_LIMIT
    time_limit_s: int = DEFAULT_SANDBOX_TIME_LIMIT_S
    memory_limit: str = DEFAULT_SANDBOX_MEMORY_LIMIT
    egress: str = "deny"  # "deny" | "allow"
    allowed_hosts: list[str] = field(default_factory=list)
    workspace_root: str = ".agent-native-workspaces"


@dataclass(slots=True)
class Settings:
    llm_provider: str = "groq"  # "groq" | "ollama"
    groq_model: str = DEFAULT_GROQ_MODEL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    groq_base_url: str | None = None

    approval_timeout_s: float = DEFAULT_APPROVAL_TIMEOUT_S
    max_calls_per_step: int = DEFAULT_MAX_CALLS_PER_STEP
    max_replans: int = DEFAULT_MAX_REPLANS
    token_budget: int = DEFAULT_TOKEN_BUDGET
    completion_budget_tokens: int = 4000

    database_url: str = DEFAULT_DATABASE_URL

    tracing_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_url: str | None = None

    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    mcp_server_commands: dict[str, list[str]] = field(
        default_factory=lambda: {
            "file": ["file-server"],
            "terminal": ["python", "-m", "terminal_server"],
            "git": ["python", "-m", "git_server"],
            "search": ["python", "-m", "search_server"],
            "memory": ["python", "-m", "memory_server"],
            "knowledge": ["python", "-m", "knowledge_server"],
        }
    )
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def groq_api_key(self) -> str | None:
        return os.environ.get("GROQ_API_KEY") or None

    def model_name(self) -> str:
        return self.groq_model if self.llm_provider == "groq" else self.ollama_model


def load_settings() -> Settings:
    """Build Settings from defaults + optional config file + environment."""
    cfg: dict[str, Any] = {}
    config_file = _find_config_file()
    if config_file is not None:
        with config_file.open("rb") as fh:
            cfg = tomllib.load(fh)
        load_dotenv(config_file.with_suffix(".env"), override=False)
    load_dotenv(override=False)

    s = Settings()
    if "llm" in cfg:
        s.llm_provider = cfg["llm"].get("provider", s.llm_provider)
        s.groq_model = cfg["llm"].get("groq_model", s.groq_model)
        s.ollama_model = cfg["llm"].get("ollama_model", s.ollama_model)
        s.groq_base_url = cfg["llm"].get("groq_base_url") or s.groq_base_url
    if "agent" in cfg:
        s.approval_timeout_s = cfg["agent"].get("approval_timeout_s", s.approval_timeout_s)
        s.max_calls_per_step = cfg["agent"].get("max_calls_per_step", s.max_calls_per_step)
        s.max_replans = cfg["agent"].get("max_replans", s.max_replans)
        s.token_budget = cfg["agent"].get("token_budget", s.token_budget)
    if "database" in cfg:
        s.database_url = cfg["database"].get("url", s.database_url)
    if "tracing" in cfg:
        s.tracing_enabled = bool(cfg["tracing"].get("enabled", s.tracing_enabled))
        s.langfuse_public_key = cfg["tracing"].get("public_key") or s.langfuse_public_key
        s.langfuse_secret_key = cfg["tracing"].get("secret_key") or s.langfuse_secret_key
        s.langfuse_url = cfg["tracing"].get("url") or s.langfuse_url
    if "sandbox" in cfg:
        sb = cfg["sandbox"]
        s.sandbox.enabled = bool(sb.get("enabled", s.sandbox.enabled))
        s.sandbox.image = sb.get("image", s.sandbox.image)
        s.sandbox.cpu_limit = str(sb.get("cpu_limit", s.sandbox.cpu_limit))
        s.sandbox.time_limit_s = int(sb.get("time_limit_s", s.sandbox.time_limit_s))
        s.sandbox.memory_limit = str(sb.get("memory_limit", s.sandbox.memory_limit))
        s.sandbox.egress = sb.get("egress", s.sandbox.egress)
        s.sandbox.allowed_hosts = list(sb.get("allowed_hosts", s.sandbox.allowed_hosts))
        s.sandbox.workspace_root = sb.get("workspace_root", s.sandbox.workspace_root)
    if "mcp" in cfg:
        for name, cmd in (cfg.get("mcp") or {}).items():
            if isinstance(cmd, list) and cmd:
                s.mcp_server_commands[name] = [str(c) for c in cmd]

    env = os.environ
    s.llm_provider = env.get("AGENT_NATIVE_LLM_PROVIDER", s.llm_provider)
    s.groq_model = env.get("AGENT_NATIVE_GROQ_MODEL", s.groq_model)
    s.ollama_model = env.get("AGENT_NATIVE_OLLAMA_MODEL", s.ollama_model)
    s.approval_timeout_s = float(env.get("AGENT_NATIVE_APPROVAL_TIMEOUT_S", s.approval_timeout_s))
    s.max_calls_per_step = int(env.get("AGENT_NATIVE_MAX_CALLS_PER_STEP", s.max_calls_per_step))
    s.max_replans = int(env.get("AGENT_NATIVE_MAX_REPLANS", s.max_replans))
    s.token_budget = int(env.get("AGENT_NATIVE_TOKEN_BUDGET", s.token_budget))
    s.database_url = env.get("AGENT_NATIVE_DATABASE_URL", s.database_url)
    s.tracing_enabled = _env_bool("AGENT_NATIVE_TRACING", s.tracing_enabled)
    s.langfuse_public_key = env.get("LANGFUSE_PUBLIC_KEY") or s.langfuse_public_key
    s.langfuse_secret_key = env.get("LANGFUSE_SECRET_KEY") or s.langfuse_secret_key
    s.langfuse_url = env.get("LANGFUSE_HOST", s.langfuse_url)

    sb_env = env.get("AGENT_NATIVE_SANDBOX_ALLOWED_HOSTS")
    if sb_env is not None:
        hosts = [h.strip() for h in sb_env.split(",") if h.strip()]
        s.sandbox.allowed_hosts = sorted(set(s.sandbox.allowed_hosts) | set(hosts))
    if env.get("AGENT_NATIVE_SANDBOX_EGRESS") is not None:
        s.sandbox.egress = env["AGENT_NATIVE_SANDBOX_EGRESS"].strip()
    if env.get("AGENT_NATIVE_SANDBOX") is not None:
        s.sandbox.enabled = _env_bool("AGENT_NATIVE_SANDBOX", s.sandbox.enabled)
    return s


def is_host_allowed(host: str, config: SandboxConfig) -> bool:
    """Egress gate for the sandbox (decision #6: deny-all unless allowlisted)."""
    if config.egress.lower() != "deny":
        return True
    host = host.strip().lower()
    return any(host == h.lower() or host.endswith("." + h.lower()) for h in config.allowed_hosts)
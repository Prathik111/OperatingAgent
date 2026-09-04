"""Build an agent-native AgentRuntime/AgentService for the API.

Lifespan owns opening/closing the native Database and wiring models/tools.
The Task-repository (TaskService) stays untouched — this is a parallel runtime.

Model wiring mirrors agent_native.main._wire_groq / _wire_ollama but without
requiring a key at startup: a missing GROQ_API_KEY just means Groq models are
not registered; the runtime still boots and send_message will return a clean
ERROR RunResult rather than crashing the process (same contract as the CLI).
MCP tools are attached lazily per-send via MCPToolProvider so the API does not
need fastmcp at import time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..config import DEFAULT_SQLITE_DATABASE_PATH

log = logging.getLogger(__name__)


def build_native_sandbox(settings: Any) -> Any | None:
    """Build the optional native container sandbox from API settings."""
    if not bool(getattr(settings, "sandbox_enabled", False)):
        return None
    try:
        from sandbox import DEFAULT_IMAGE, ContainerSandbox

        return ContainerSandbox(
            image=str(getattr(settings, "sandbox_image", "") or DEFAULT_IMAGE),
            network=False,
        )
    except (ImportError, TypeError) as exc:
        log.warning("Native sandbox unavailable: %s", exc)
        return None


def build_native_database(settings: Any) -> tuple[Any, Any]:
    """Return (Database, pool_or_None) for the native track.

    Uses the configured SQLite path for desktop persistence, reuses
    ``DATABASE_URL`` for PostgreSQL, and falls back to ``MemoryDatabase`` only
    when explicitly configured. Returns a pool handle only for the postgres
    branch so lifespan can await open/close.
    """
    database_url = getattr(settings, "database_url", None)
    backend = (getattr(settings, "repository_backend", "memory") or "memory").lower()

    # Explicit postgres request must have a DSN
    if backend == "postgres" and not database_url:
        raise ValueError("repository_backend is 'postgres' but DATABASE_URL is not set")

    if backend in {"sqlite", "file", "file-based", "file_based"}:
        from agent_native.sqlite import SQLiteDatabase

        return SQLiteDatabase(
            getattr(settings, "sqlite_database_path", str(DEFAULT_SQLITE_DATABASE_PATH))
        ), None

    # If a real Postgres DSN is present, use the native PostgresDatabase
    # even when repository_backend is still 'memory' for the Task API — the
    # two stores are independent and native should be durable when it can be.
    if database_url:
        try:
            from agent_native.postgres import PostgresDatabase

            # PostgresDatabase manages its own asyncpg pool internally;
            # we return the instance itself as the 'pool' so lifespan can
            # await .connect() / .close() without a second pool type.
            db = PostgresDatabase(database_url)
            return db, db  # db doubles as openable/closeable
        except Exception as exc:  # noqa: BLE001 - native Postgres is optional
            log.warning("Failed to init native PostgresDatabase, falling back to memory: %s", exc)

    from agent_native.database import MemoryDatabase

    return MemoryDatabase(), None


def wire_native_models(runtime: Any, settings: Any | None = None) -> list[str]:
    """Register Groq/Ollama providers onto the runtime's ModelRegistry.

    Best-effort, no raise: missing keys or optional deps just mean that model
    is not available, which the loop turns into a clean ERROR RunResult.
    Returns the list of model names registered.
    """
    registered: list[str] = []

    configured_provider = str(
        getattr(settings, "llm_provider", "") or os.getenv("LLM_PROVIDER", "")
    ).strip().lower()
    configured_model = str(
        getattr(settings, "llm_model", "") or os.getenv("LLM_MODEL", "")
    ).strip()

    # Groq
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            from agent_native.models.base import Model
            from agent_native.models.groq_model import GROQ_MODELS, Groq

            groq = Groq()
            runtime.models.register_provider("groq", groq)
            # Register the canonical GROQ_MODELS plus any LLM_MODEL override
            for short_name, model in GROQ_MODELS.items():
                try:
                    runtime.models.register_model(short_name, model)
                    registered.append(short_name)
                    # also register under the full model_id for direct lookup
                    if model.model_id not in registered:
                        runtime.models.register_model(model.model_id, model)
                        registered.append(model.model_id)
                except Exception as exc:  # noqa: BLE001 - one bad alias must not block others
                    log.debug("Skipping Groq model alias %s: %s", short_name, exc)
                    continue
            # LLM_MODEL env may be a custom Groq model id
            custom = (
                configured_model or os.getenv("GROQ_MODEL") or ""
                if configured_provider == "groq"
                else ""
            )
            # Alias the legacy default gpt-oss-120b to a real Groq model
            if custom == "gpt-oss-120b":
                custom = ""
            if custom and custom not in GROQ_MODELS and custom not in registered:
                model = Model(provider="groq", model_id=custom, context_size=128_000, max_output=8192)
                try:
                    runtime.models.register_model(custom, model)
                    registered.append(custom)
                except Exception as exc:  # noqa: BLE001 - custom model is optional
                    log.debug("Skipping custom Groq model %s: %s", custom, exc)
            # Default build agent's model: gpt-oss-120b is the default in AgentConfig
            # Map it to llama-3.3-70b if not otherwise registered so send_message doesn't KeyError
            if "gpt-oss-120b" not in registered:
                fallback = GROQ_MODELS.get("llama-3.3-70b")
                if fallback is not None:
                    try:
                        runtime.models.register_model("gpt-oss-120b", fallback)
                        registered.append("gpt-oss-120b")
                    except Exception as exc:  # noqa: BLE001 - legacy alias is optional
                        log.debug("Skipping native default model alias: %s", exc)
        except Exception as exc:  # noqa: BLE001 - provider wiring is optional
            log.debug("Groq wiring skipped: %s", exc)
    else:
        # No key: still register a placeholder mapping so list_models shows intent?
        # Don't register a provider — loop will error cleanly on send_message.
        log.debug("GROQ_API_KEY not set; native Groq models not registered")

    # Ollama (optional)
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        from agent_native.models.base import Model, ToolFormat
        from agent_native.models.ollama_model import Ollama

        ollama = Ollama(host=ollama_host)
        runtime.models.register_provider("ollama", ollama)
        # Register a sensible default if not already present
        defaults = [("qwen3.5:4b-q4_K_M", "ollama"), ("llama3.1", "ollama")]
        if configured_provider == "ollama" and configured_model:
            defaults.insert(0, (configured_model, "ollama"))
        for name, provider in defaults:
            if name not in registered:
                try:
                    m = Model(provider=provider, model_id=name, context_size=8192, max_output=2048, tool_format=ToolFormat.NATIVE)
                    runtime.models.register_model(name, m)
                    registered.append(name)
                except Exception as exc:  # noqa: BLE001 - one optional model must not block others
                    log.debug("Skipping Ollama model %s: %s", name, exc)
    except Exception as exc:  # noqa: BLE001 - provider wiring is optional
        log.debug("Ollama wiring skipped: %s", exc)

    return registered


async def attach_mcp_tools(runtime: Any, working_directory: str = ".") -> list[Any]:
    """Attach the gateway's MCP tools to the runtime for one working directory.

    Called lazily on first send_message per working_directory; attaches globally
    so subsequent sessions in same folder reuse the same tools. Returns attached tools.
    """
    # Check if MCP already attached (runtime.tools has >6 native tools)
    native_names = {"remember", "recall", "plan", "invoke_skill", "delegate", "fan_out"}
    existing = {t.definition.full_name for t in runtime.tools.all()}
    if len(existing - native_names) > 0:
        return []  # already attached

    try:
        from agent_native.tools.mcp_bridge import MCPToolProvider
    except Exception as exc:  # noqa: BLE001 - MCP bridge is optional
        log.debug("MCP bridge not available: %s", exc)
        return []

    try:
        provider = MCPToolProvider()
        root = str(Path(working_directory).expanduser().resolve()) if working_directory and working_directory != "." else str(Path.cwd())
        # Ensure root exists — fallback to CWD if not
        if not Path(root).is_dir():
            root = str(Path.cwd())
        tools = await provider.connect(root=root)
        for t in tools:
            try:
                runtime.tools.register(t)
            except Exception as exc:  # noqa: BLE001 - duplicate/incompatible tools are skippable
                log.debug("Skipping MCP tool registration: %s", exc)
                continue
        # Keep provider alive on runtime so it can be closed on shutdown
        # Store on a private attr to avoid polluting the public API
        if not hasattr(runtime, "_mcp_providers"):
            runtime._mcp_providers = []  # type: ignore[attr-defined]
        runtime._mcp_providers.append(provider)  # type: ignore[attr-defined]
        log.info("Attached %d MCP tools for %r", len(tools), root)
        return tools
    except Exception as exc:  # noqa: BLE001 - MCP connection is optional
        log.warning("Failed to attach MCP tools for %r: %s", working_directory, exc)
        return []

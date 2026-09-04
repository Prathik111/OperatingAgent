"""Support for the opt-in live tiers (``tests/integration/live``, ``tests/e2e/live``).

The default suite is hermetic: ``tests/conftest.py`` strips provider and Langfuse
credentials out of the environment before collection, so no unit, integration or
e2e test can reach a real service. These helpers are the *only* way credentials
come back, and they do it per-test through ``monkeypatch`` so the restoration is
reverted the moment the test ends. Credentials are never re-exported at import
time, which is what keeps the hermetic tiers hermetic even in the same session.

Credentials are read from ``tests.conftest.CLEARED_CREDENTIALS`` (whatever the
bootstrap stripped — covers both ``.env.test`` and shell-exported values) with
``.env.test`` consulted as a fallback.

Two environment facts, both established empirically against the project's own
accounts and easy to get wrong:

* The Langfuse keys authenticate against the **EU** region. ``.env.test`` points
  ``LANGFUSE_HOST`` at localhost and holds the cloud URL in ``LANGFUSE_RELEASE``
  (they are swapped), so the host is forced here rather than trusted.
* The Groq account has ``openai/gpt-oss-20b`` but **not** ``llama-3.3-70b-versatile``.
  The planner requests ``method="json_schema"`` structured output, so whichever
  model is used must support it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
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

#: Set to "1" to let the live tiers run. Already set in ``.env.test``.
LIVE_FLAG = "OPERATING_AGENT_ENABLE_LIVE_TESTS"

#: Region the project's Langfuse keys belong to.
LANGFUSE_HOST = "https://cloud.langfuse.com"

#: A Groq model this account can reach that supports JSON-schema structured output.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_TEST = REPO_ROOT / ".env.test"

_REQUIRED = ("GROQ_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def _credentials() -> dict[str, str]:
    """Credentials the bootstrap cleared, falling back to ``.env.test``."""
    from tests.conftest import CLEARED_CREDENTIALS

    values: dict[str, str] = {}
    if ENV_TEST.is_file():
        try:
            from dotenv import dotenv_values
        except ModuleNotFoundError:  # python-dotenv is a dev dep; tolerate absence
            pass
        else:
            values.update({k: v for k, v in dotenv_values(ENV_TEST).items() if v})
    # Anything the bootstrap stripped wins: it reflects the effective environment.
    values.update(CLEARED_CREDENTIALS)
    return values


@pytest.fixture(autouse=True)
def live_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Restore real credentials for one live test, then revert.

    Autouse within whichever ``live/`` directory imports it, so every test in
    that directory is gated and configured identically. Skips (never fails) when
    the flag is off or a key is missing — a developer without credentials should
    get a green run with skips, not a wall of errors.
    """
    if os.environ.get(LIVE_FLAG) != "1":
        pytest.skip(f"live tests are opt-in: set {LIVE_FLAG}=1 (see .env.test)")

    credentials = _credentials()
    missing = [name for name in _REQUIRED if not credentials.get(name)]
    if missing:
        pytest.skip(f"missing live credentials: {', '.join(missing)}")

    for name in _REQUIRED:
        monkeypatch.setenv(name, credentials[name])

    # Forced, not inherited: see the module docstring on the swapped host/release.
    monkeypatch.setenv("LANGFUSE_HOST", LANGFUSE_HOST)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_RELEASE", "pytest-live")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "test")

    # observability.client caches its Langfuse instance in module globals; the
    # autouse reset_observability_singletons fixture clears it around every test,
    # but clear it again *after* the credentials land so the first get_client()
    # inside the test re-reads the now-populated environment.
    from observability import client as observability_client

    observability_client._client = None
    observability_client._initialised = False

    return credentials


def groq_model() -> str:
    """The model live tests should use (override with ``GROQ_MODEL``)."""
    return os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL


class InlinePromptManager:
    """Real system prompts, supplied inline.

    The repo ships no ``prompts/`` directory, so the real ``PromptManager`` would
    raise ``FileNotFoundError``. These are genuine prompts — the LLM behaviour
    under test is live, only the *storage* of the prompt text is inlined.
    """

    def planner(self) -> str:
        return (
            "You are a planning agent. Break the user's goal into a short, "
            "ordered list of concrete steps. When a step should call a tool, "
            "set tool_name to one of the available tools and fill 'arguments' "
            "to match that tool's inputs exactly. Use a final reasoning step "
            "with tool_name=null to summarise. Keep the plan minimal."
        )

    def verifier(self) -> str:
        return (
            "You are a verification agent. Given a step's intended outcome and "
            "the output it produced, decide whether the step succeeded. Respond "
            "success=true only if the output clearly satisfies the step's intent."
        )

    def responder(self) -> str:
        return (
            "You are a helpful assistant. Using the results of the completed "
            "steps, write a concise final answer to the user's original goal."
        )


def build_live_config(
    *,
    model: str | None = None,
    tracing_enabled: bool = True,
    require_verification: bool = True,
    max_iterations: int = 8,
    timeout_seconds: int = 60,
    retry_attempts: int = 1,
    **overrides: Any,
) -> AgentConfig:
    """An ``AgentConfig`` pointed at the real Groq provider and real Langfuse.

    ``require_human_approval`` is always False: the executor's ``interrupt()``
    gate would block a graph nobody is watching.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:  # pragma: no cover - the fixture skips before this
        pytest.skip("GROQ_API_KEY is not set")

    return AgentConfig(
        llm=LLMConfig(
            provider="groq",
            model=model or groq_model(),
            api_key=api_key,
            temperature=0.0,
            max_tokens=1024,
        ),
        execution=ExecutionConfig(
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            enable_checkpoints=True,
        ),
        sandbox=SandboxConfig(),
        permissions=ToolPermissionConfig(),
        checkpoint=CheckpointConfig(backend="memory"),
        tracing=TracingConfig(
            enabled=tracing_enabled, provider="langfuse", project_name="OperatingAgent"
        ),
        behaviour=BehaviourConfig(
            require_verification=require_verification,
            require_human_approval=False,
        ),
        # Unused: live tests inject InlinePromptManager. Required by the dataclass.
        prompts=PromptConfig(
            planner_prompt=Path("prompts/planner.txt"),
            verifier_prompt=Path("prompts/verifier.txt"),
            responder_prompt=Path("prompts/responder.txt"),
        ),
        metadata=MetadataConfig(),
        **overrides,
    )

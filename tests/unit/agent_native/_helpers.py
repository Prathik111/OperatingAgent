"""Shared helpers for the end-to-end tests.

Not a test module itself (the name doesn't start with test_), just a couple of
constructors so the loop tests read like sentences. No pytest fixtures on
purpose, so the same tests run under a plain stdlib runner too.

These helpers build a runtime backed by the **real** Groq model, so the loop tests
make live API calls. That means they need `groq` installed, a `GROQ_API_KEY`, and
network access. `live_groq_available()` reports whether all three are plausible so
a test can skip instead of failing on a machine that can't reach Groq.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from agent_native.config import AgentConfig
from agent_native.loop import Limits
from agent_native.models.base import Model
from agent_native.models.groq_model import GROQ_MODELS, Groq
from agent_native.permissions import PermissionDuration
from agent_native.service import AgentRuntime, AgentService

from tests._fake_tools import default_tools as fake_default_tools


class Skip(Exception):
    """Raised by a test that can't run here (no key, no groq, no network)."""


def live_groq_available() -> str | None:
    """Return None if a live Groq run is possible, else a reason string."""
    if os.getenv("OPERATING_AGENT_ENABLE_LIVE_TESTS") != "1":
        return "set OPERATING_AGENT_ENABLE_LIVE_TESTS=1 to enable paid live tests"
    try:
        import groq  # noqa: F401
    except ImportError:
        return "the 'groq' package is not installed"
    if not Groq().has_key:
        return "GROQ_API_KEY is not set"
    return None


def require_live_groq() -> None:
    """Skip the calling test unless a real Groq call can be made.

    Uses pytest's own skip when running under pytest, so the suite reports these
    as skipped rather than failed; falls back to `Skip` for the stdlib runner.
    """
    reason = live_groq_available()
    if not reason:
        return
    try:
        import pytest
    except ImportError:
        raise Skip(reason) from None
    pytest.skip(reason)


def make_runtime(
    workdir: str = ".",
    model_name: str = "llama-3.3-70b",
    max_turns: int = 8,
    temperature: float = 0.0,
):
    """A runtime wired to the real Groq model, with the fake file tools registered."""
    config = AgentConfig(
        name="build", model=model_name, max_turns=max_turns, temperature=temperature
    )
    runtime = AgentRuntime(agents=[config])
    for tool in fake_default_tools():
        runtime.tools.register(tool)

    runtime.models.register_provider("groq", Groq())
    model = GROQ_MODELS.get(model_name) or Model(
        provider="groq", model_id=model_name, context_size=128_000, max_output=8192
    )
    runtime.models.register_model(model_name, model)

    service = AgentService(runtime)
    return runtime, service


async def run_with_auto_permissions(
    service: AgentService,
    session_id: str,
    text: str,
    answer: bool = True,
    duration: Any = PermissionDuration.SESSION,
    limits: Limits | None = None,
):
    """Run send_message while a background task answers any permission prompt.

    The loop awaits the user on a permission ask; this drains those prompts
    concurrently so a test can drive a full run without a UI.
    """
    done = asyncio.Event()

    async def approver() -> None:
        while not done.is_set():
            for request in service.pending_permissions():
                await service.resolve_permission(request.call_id, answer, duration)
            await asyncio.sleep(0.005)

    task = asyncio.create_task(approver())
    try:
        return await service.send_message(session_id, text, limits=limits)
    finally:
        done.set()
        await task

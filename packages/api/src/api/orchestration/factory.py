"""Build the per-track orchestrator map, degrading rather than crashing.

The native track is always the local stub. The LangGraph track is the real
``LangGraphAgent`` — but its constructor eagerly builds a ``ModelProvider``,
which raises for an unsupported/misconfigured provider (and its model
integration may be absent). So construction is **guarded**: on any failure we
register an ``UnavailableOrchestrator`` that emits an ``error`` event and returns
a ``FAILED`` result. The app therefore boots with no LLM credentials — the
native track works, and the langgraph track fails cleanly per-run instead of
taking the process down at startup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent
from common.interfaces import IAgentOrchestrator

from .native_stub import EventCallback, NativeStubOrchestrator, emit_event

if TYPE_CHECKING:
    from ..config import ApiSettings

log = logging.getLogger(__name__)


class UnavailableOrchestrator:
    """Stands in for a track that could not be constructed; every run fails cleanly."""

    def __init__(self, track: str, reason: str) -> None:
        self._track = track
        self._reason = reason

    async def run(
        self, task: AgentTask, on_event: EventCallback = None
    ) -> AgentRunResult:
        message = f"orchestrator for track '{self._track}' is unavailable: {self._reason}"
        await emit_event(on_event, AgentEvent(type="error", payload={"error": message}))
        await emit_event(
            on_event,
            AgentEvent(
                type="finished",
                payload={"status": RunStatus.FAILED.value, "trace_id": None},
            ),
        )
        return AgentRunResult(
            status=RunStatus.FAILED,
            output=None,
            duration_ms=0.0,
            llm_calls=0,
            tool_calls=0,
            total_tokens=0,
            metadata={"error": message},
        )


def build_orchestrators(
    settings: "ApiSettings",
) -> dict[AgentTrack, IAgentOrchestrator]:
    orchestrators: dict[AgentTrack, IAgentOrchestrator] = {
        AgentTrack.NATIVE: NativeStubOrchestrator(),
    }

    try:
        from agent_langgraph.orchestrator.langgraph_agent import LangGraphAgent

        config = settings.build_agent_config(AgentTrack.LANGGRAPH)
        orchestrators[AgentTrack.LANGGRAPH] = LangGraphAgent(config)
    except Exception as exc:  # eager ModelProvider / missing integration / etc.
        log.warning(
            "LangGraph orchestrator unavailable, registering degraded stub: %s", exc
        )
        orchestrators[AgentTrack.LANGGRAPH] = UnavailableOrchestrator(
            AgentTrack.LANGGRAPH.value, str(exc)
        )

    return orchestrators

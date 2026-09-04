"""Build the per-track ``IAgentOrchestrator`` map, degrading rather than crashing.

The native track delegates to the real ``agent_native`` service. The LangGraph track is the real
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
from importlib import import_module
from typing import TYPE_CHECKING

from common.agent import AgentRunResult, AgentTask
from common.approvals import ApprovalHandler
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent
from common.interfaces import IAgentOrchestrator

from .native import EventCallback, NativeAgentOrchestrator, _emit

if TYPE_CHECKING:
    from ..config import ApiSettings

log = logging.getLogger(__name__)


class UnavailableOrchestrator(IAgentOrchestrator):
    """Stands in for a track that could not be constructed; every run fails cleanly."""

    def __init__(self, track: str, reason: str) -> None:
        self._track = track
        self._reason = reason

    async def run(
        self, task: AgentTask, on_event: EventCallback = None
    ) -> AgentRunResult:
        message = f"orchestrator for track '{self._track}' is unavailable: {self._reason}"
        await _emit(on_event, AgentEvent(type="error", payload={"error": message}))
        await _emit(
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

    async def aclose(self) -> None:
        """The degraded orchestrator owns no external resources."""


def build_orchestrators(
    settings: ApiSettings,
    *,
    approval_handler: ApprovalHandler | None = None,
    native_service: object | None = None,
) -> dict[AgentTrack, IAgentOrchestrator]:
    orchestrators: dict[AgentTrack, IAgentOrchestrator] = {
        AgentTrack.NATIVE: (
            NativeAgentOrchestrator(native_service)
            if native_service is not None
            else UnavailableOrchestrator(
                AgentTrack.NATIVE.value, "native service is not initialized"
            )
        ),
    }

    # Configuration is application state, not an optional integration. Validate
    # it before entering the degradation boundary so invalid settings fail at
    # startup instead of being hidden behind an unavailable track.
    config = settings.build_agent_config(AgentTrack.LANGGRAPH)

    try:
        module = import_module("agent_langgraph.orchestrator.langgraph_agent")
        LangGraphAgent = module.LangGraphAgent

        orchestrators[AgentTrack.LANGGRAPH] = LangGraphAgent(
            config,
            approval_handler=approval_handler,
            mcp_gateway_command=settings.mcp_gateway_command,
            mcp_gateway_args=list(settings.mcp_gateway_args),
        )
    except Exception as exc:
        log.warning(
            "LangGraph orchestrator unavailable, registering degraded stub: %s", exc
        )
        orchestrators[AgentTrack.LANGGRAPH] = UnavailableOrchestrator(
            AgentTrack.LANGGRAPH.value, str(exc)
        )

    return orchestrators

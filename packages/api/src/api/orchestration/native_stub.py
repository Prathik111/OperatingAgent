"""A placeholder native-track orchestrator.

The native track is not implemented yet, but the API dispatches to both tracks,
so this stand-in satisfies ``IAgentOrchestrator`` and emits the same event shape
the LangGraph track does (``state`` then ``finished``). It lets a client submit
``track=native`` and get a complete, streamed, ``COMPLETED`` run with no LLM
credentials — useful as a smoke path and as the hermetic orchestrator in tests.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from common.agent import AgentRunResult, AgentTask
from common.enums import RunStatus
from common.events import AgentEvent

log = logging.getLogger(__name__)

EventCallback = Callable[[AgentEvent], Awaitable[None] | None] | None


async def emit_event(on_event: EventCallback, event: AgentEvent) -> None:
    """Deliver an event tolerating sync/async callbacks and swallowing listener
    errors — the same contract as ``LangGraphAgent._emit`` so both tracks behave
    identically for a subscriber."""
    if on_event is None:
        return
    try:
        outcome = on_event(event)
        if outcome is not None and hasattr(outcome, "__await__"):
            await outcome
    except Exception as exc:  # noqa: BLE001 - callback isolation boundary
        log.warning("event callback raised: %s", exc)


class NativeStubOrchestrator:
    """``IAgentOrchestrator`` that echoes the goal and completes immediately."""

    async def run(
        self, task: AgentTask, on_event: EventCallback = None
    ) -> AgentRunResult:
        await emit_event(
            on_event,
            AgentEvent(type="state", payload={"status": "running", "current_step": 0}),
        )
        output = f"[native-stub] received goal: {task.goal}"
        await emit_event(
            on_event,
            AgentEvent(
                type="finished",
                payload={"status": RunStatus.COMPLETED.value, "trace_id": None},
            ),
        )
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output=output,
            duration_ms=0.0,
            llm_calls=0,
            tool_calls=0,
            total_tokens=0,
        )

    async def aclose(self) -> None:
        """The native stub owns no external resources."""

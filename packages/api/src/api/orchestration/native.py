"""Adapt the session-oriented native agent to the shared Task API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from common.agent import AgentRunResult, AgentTask
from common.enums import RunStatus
from common.events import AgentEvent
from common.interfaces import IAgentOrchestrator

log = logging.getLogger(__name__)
EventCallback = Callable[[AgentEvent], Awaitable[None] | None] | None


async def _emit(on_event: EventCallback, event: AgentEvent) -> None:
    if on_event is None:
        return
    try:
        outcome = on_event(event)
        if outcome is not None and hasattr(outcome, "__await__"):
            await outcome
    except Exception as exc:  # noqa: BLE001 - callbacks must not break a run
        log.warning("native event callback raised: %s", exc)


class NativeAgentOrchestrator(IAgentOrchestrator):
    """Run shared tasks through ``agent_native.service.AgentService``."""

    def __init__(self, service: Any) -> None:
        if service is None:
            raise ValueError("native service is required")
        self._service = service

    async def _session_for(self, task: AgentTask) -> Any:
        database = self._service.runtime.database
        session = await database.get_session(task.thread_id)
        if session is not None:
            return session

        metadata = task.metadata or {}
        workspace = str(
            metadata.get("workspace") or metadata.get("working_directory") or "."
        )
        return await self._service.create_session(
            agent=str(metadata.get("agent") or "build"),
            title=str(metadata.get("title") or ""),
            working_directory=workspace,
            session_id=task.thread_id,
        )

    @staticmethod
    def _event_type(native_type: str) -> str:
        return {
            "run_finished": "finished",
            "message_added": "state",
        }.get(native_type, native_type)

    @staticmethod
    def _event_payload(event: Any) -> dict[str, Any]:
        payload = dict(getattr(event, "data", {}) or {})
        native_type = str(getattr(event, "type", ""))
        payload.update(
            {
                "native_event_type": native_type,
                "native_session_id": str(getattr(event, "session_id", "")),
                "native_run_id": str(getattr(event, "run_id", "") or ""),
                "native_sequence": int(getattr(event, "sequence", 0) or 0),
            }
        )
        if native_type == "run_finished" and "output" not in payload:
            payload["output"] = payload.get("final_text")
        return payload

    async def run(
        self, task: AgentTask, on_event: EventCallback = None
    ) -> AgentRunResult:
        session = await self._session_for(task)
        database = self._service.runtime.database
        previous_events = await database.load_events(session.id, after_sequence=0)
        tip = max(
            (int(getattr(event, "sequence", 0) or 0) for event in previous_events),
            default=0,
        )
        tool_calls = 0

        async def forward_events() -> None:
            nonlocal tool_calls
            async for event in self._service.subscribe(session.id, from_sequence=tip):
                native_type = str(getattr(event, "type", ""))
                if native_type == "tool_started":
                    tool_calls += 1
                await _emit(
                    on_event,
                    AgentEvent(
                        type=self._event_type(native_type),
                        payload=self._event_payload(event),
                    ),
                )
                run_id = str(getattr(event, "run_id", "") or "")
                if native_type == "run_finished" and "/" not in run_id:
                    return

        forwarder = asyncio.create_task(forward_events())
        await asyncio.sleep(0)
        try:
            try:
                from ..native.runtime import attach_mcp_tools

                await asyncio.wait_for(
                    attach_mcp_tools(
                        self._service.runtime,
                        working_directory=getattr(session, "working_directory", "."),
                    ),
                    timeout=5,
                )
            except Exception as exc:  # noqa: BLE001 - MCP integration is optional
                log.debug("native MCP attachment skipped: %s", exc)
            result = await self._service.send_message(session.id, task.goal)
        finally:
            if not forwarder.done():
                try:
                    await asyncio.wait_for(forwarder, timeout=2)
                except (TimeoutError, asyncio.CancelledError):
                    forwarder.cancel()
                    await asyncio.gather(forwarder, return_exceptions=True)
                except Exception as exc:  # noqa: BLE001 - forwarding is best effort
                    log.debug("native event forwarder failed during cleanup: %s", exc)
            else:
                try:
                    forwarder.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001 - forwarding is best effort
                    log.debug("native event forwarder failed: %s", exc)

        native_status = str(
            getattr(getattr(result, "status", None), "value", "") or ""
        )
        status = {
            "finished": RunStatus.COMPLETED,
            "error": RunStatus.FAILED,
            "cancelled": RunStatus.INTERRUPTED,
            "limit_reached": RunStatus.INTERRUPTED,
        }.get(native_status, RunStatus.FAILED)
        usage = getattr(result, "usage", None)
        metadata = {
            "native_run_id": str(getattr(result, "run_id", "") or ""),
            "native_status": native_status,
            "model": str(getattr(result, "model", "") or ""),
            "trace_id": str(getattr(result, "trace_id", "") or ""),
        }
        error = str(getattr(result, "error", "") or "")
        if error:
            metadata["error"] = error
        return AgentRunResult(
            status=status,
            output=str(getattr(result, "final_text", "") or "") or None,
            duration_ms=float(getattr(result, "duration_seconds", 0.0) or 0.0) * 1000,
            llm_calls=int(getattr(result, "turns", 0) or 0),
            tool_calls=tool_calls,
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            cost=float(getattr(result, "cost_usd", 0.0) or 0.0),
            metadata=metadata,
        )

    async def aclose(self) -> None:
        """The application lifespan owns the native runtime resources."""

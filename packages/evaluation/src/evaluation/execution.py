"""Execution helpers for running evaluation suites against real tracks."""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from agent_native.config import AgentConfig as NativeAgentConfig
from agent_native.service import AgentRuntime, AgentService
from api.config import ApiSettings
from api.environment import load_environment
from api.native.runtime import (
    build_native_database,
    build_native_sandbox,
    wire_native_models,
)
from api.orchestration.factory import build_orchestrators
from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack

from .runner import EvaluationResult, EvaluationRunner
from .suite import EvaluationSuite

log = logging.getLogger(__name__)


async def _maybe_await(value: object) -> None:
    """Await a dynamically-discovered cleanup result when it is awaitable."""
    if inspect.isawaitable(value):
        await value


async def _close_resource(resource: object | None, *method_names: str) -> None:
    """Close one optional resource without masking the original failure."""
    if resource is None:
        return
    for method_name in method_names:
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            await _maybe_await(method())
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the run
            log.debug("evaluation resource cleanup failed: %s", exc)
        return


@dataclass(slots=True)
class EvaluationEnvironment:
    settings: ApiSettings
    orchestrators: dict[AgentTrack, Any]
    native_runtime: AgentRuntime | None = None
    native_pool: Any | None = None

    async def aclose(self) -> None:
        for orchestrator in self.orchestrators.values():
            await _close_resource(orchestrator, "aclose", "close")
        if self.native_runtime is not None:
            for provider in getattr(self.native_runtime, "_mcp_providers", []) or []:
                await _close_resource(provider, "aclose", "close")
            await _close_resource(self.native_runtime.events, "aclose", "close")
            await _close_resource(self.native_runtime.database, "aclose", "close")
            sandbox = getattr(self.native_runtime, "sandbox", None)
            await _close_resource(sandbox, "aclose", "close", "stop_all")
            try:
                list(self.native_runtime.monitoring.shutdown())
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask the run
                log.debug("native monitoring shutdown failed: %s", exc)
        elif self.native_pool is not None:
            await _close_resource(self.native_pool, "aclose", "close", "stop_all")


@asynccontextmanager
async def open_evaluation_environment(settings: ApiSettings | None = None):
    if settings is None:
        load_environment()
    resolved_settings = settings or ApiSettings.from_env()
    native_db, native_pool = build_native_database(resolved_settings)
    native_runtime: AgentRuntime | None = None
    native_sandbox: Any | None = None
    orchestrators: dict[AgentTrack, Any] = {}
    try:
        if native_pool is not None and hasattr(native_pool, "connect"):
            await native_pool.connect()

        native_config = NativeAgentConfig(
            name="build",
            model=resolved_settings.llm_model,
            max_turns=resolved_settings.execution_max_iterations,
            temperature=resolved_settings.llm_temperature,
        )
        native_sandbox = build_native_sandbox(resolved_settings)
        native_runtime = AgentRuntime(
            database=native_db,
            agents=[native_config],
            sandbox=native_sandbox,
        )
        wire_native_models(native_runtime, settings=resolved_settings)
        native_service = AgentService(native_runtime)
        orchestrators = build_orchestrators(
            resolved_settings,
            native_service=native_service,
        )
        environment = EvaluationEnvironment(
            settings=resolved_settings,
            orchestrators=orchestrators,
            native_runtime=native_runtime,
            native_pool=native_pool,
        )
        yield environment
    finally:
        if native_runtime is not None:
            await EvaluationEnvironment(
                settings=resolved_settings,
                orchestrators=orchestrators,
                native_runtime=native_runtime,
                native_pool=native_pool,
            ).aclose()
        else:
            await _close_resource(native_sandbox, "aclose", "close", "stop_all")
            await _close_resource(native_pool, "aclose", "close", "stop_all")


async def run_suite_for_tracks(
    suite: EvaluationSuite,
    environment: EvaluationEnvironment,
    tracks: tuple[AgentTrack, ...] = (AgentTrack.NATIVE, AgentTrack.LANGGRAPH),
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for track in tracks:
        orchestrator = environment.orchestrators[track]
        runner = EvaluationRunner(suite)

        async def execute(task: AgentTask, selected: Any = orchestrator) -> AgentRunResult:
            return await _run_task(selected, task)

        results.extend(
            await runner.run(
                track,
                execute,
                session_prefix=f"eval-{track.value}-",
            )
        )
    return results


async def run_suite(
    suite: EvaluationSuite,
    settings: ApiSettings | None = None,
    tracks: tuple[AgentTrack, ...] = (AgentTrack.NATIVE, AgentTrack.LANGGRAPH),
) -> list[EvaluationResult]:
    async with open_evaluation_environment(settings) as environment:
        return await run_suite_for_tracks(suite, environment, tracks)


async def _run_task(orchestrator: Any, task: Any) -> AgentRunResult:
    return await orchestrator.run(task)

"""Generic execution hooks for evaluation suites."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack

from .suite import EvaluationCase, EvaluationSuite

CaseRunner = Callable[[AgentTask], Awaitable[AgentRunResult]]


@dataclass(slots=True)
class EvaluationResult:
    case_id: str
    track: str
    passed: bool
    output: str = ""
    error: str = ""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        return cls(
            case_id=str(data.get("case_id", "")),
            track=str(data.get("track", "")),
            passed=bool(data.get("passed", False)),
            output=str(data.get("output", "")),
            error=str(data.get("error", "")),
            turns=int(data.get("turns", 0) or 0),
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cached_tokens=int(data.get("cached_tokens", 0) or 0),
            reasoning_tokens=int(data.get("reasoning_tokens", 0) or 0),
            cost=float(data.get("cost", 0.0) or 0.0),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


class EvaluationRunner:
    def __init__(self, suite: EvaluationSuite) -> None:
        self._suite = suite

    async def run(
        self,
        track: AgentTrack,
        run_case: CaseRunner,
        session_prefix: str = "eval-",
    ) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for case in self._suite.cases:
            results.append(
                await self.run_case(case, track, run_case, session_prefix=session_prefix)
            )
        return results

    async def run_case(
        self,
        case: EvaluationCase,
        track: AgentTrack,
        run_case: CaseRunner,
        session_prefix: str = "eval-",
    ) -> EvaluationResult:
        task = AgentTask(
            id=f"eval-{case.id}",
            goal=case.goal,
            thread_id=f"{session_prefix}{track.value}-{case.id}",
            track=track,
            metadata={
                **dict(case.metadata),
                "working_directory": case.working_directory,
            },
        )
        started = time.perf_counter()
        try:
            result = await run_case(task)
        except Exception as exc:  # noqa: BLE001 - one case must not abort a suite
            return EvaluationResult(
                case_id=case.id,
                track=track.value,
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=round(time.perf_counter() - started, 3),
                metadata=dict(task.metadata),
            )
        output = result.output or ""
        expected = case.expected_output_contains.lower()
        passed = bool(output) and (not expected or expected in output.lower())
        return EvaluationResult(
            case_id=case.id,
            track=track.value,
            passed=passed and result.status.value == "completed",
            output=output,
            error=str(result.metadata.get("error", "") if result.metadata else ""),
            turns=int(result.llm_calls or 0),
            input_tokens=int(result.total_tokens or 0),
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            cost=float(result.cost or 0.0),
            duration_seconds=round(time.perf_counter() - started, 3),
            metadata=dict(result.metadata or {}),
        )


async def run_case(case: EvaluationCase, track: AgentTrack, run_case: CaseRunner) -> EvaluationResult:
    return await EvaluationRunner(EvaluationSuite(id="single", cases=(case,))).run_case(
        case,
        track,
        run_case,
    )

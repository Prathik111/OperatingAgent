"""Generic execution hooks for evaluation suites."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack
from common.judging import expected_output_check, judge_output

from .scoring import LLMJudge
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
    checks: list[dict[str, Any]] = field(default_factory=list)
    judge: dict[str, Any] | None = None
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
            checks=list(data.get("checks", []) or []),
            judge=dict(data["judge"]) if data.get("judge") else None,
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
        judge: LLMJudge | None = None,
    ) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for case in self._suite.cases:
            results.append(
                await self.run_case(
                    case, track, run_case, session_prefix=session_prefix, judge=judge
                )
            )
        return results

    async def run_case(
        self,
        case: EvaluationCase,
        track: AgentTrack,
        run_case: CaseRunner,
        session_prefix: str = "eval-",
        judge: LLMJudge | None = None,
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
        completed = result.status.value == "completed"
        # Deterministic judgement stays the headline: completed run, non-empty
        # output, and every named check passing. The LLM judge, when present,
        # never influences `passed` — its verdict is recorded beside it.
        checks = _case_checks(case)
        judgment = judge_output(output, checks) if completed else None
        passed = bool(output) and judgment is not None and judgment.passed
        judge_payload: dict[str, Any] | None = None
        if judge is not None and completed and output.strip():
            verdict = await judge.judge(goal=case.goal, output=output, checks=checks)
            judge_payload = verdict.to_dict()
        error = str(result.metadata.get("error", "") if result.metadata else "")
        if not passed and judgment is not None and judgment.failure_summary():
            detail = judgment.failure_summary()
            error = f"{error}; {detail}" if error else detail
        return EvaluationResult(
            case_id=case.id,
            track=track.value,
            passed=passed,
            output=output,
            error=error,
            turns=int(result.llm_calls or 0),
            input_tokens=int(result.total_tokens or 0),
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            cost=float(result.cost or 0.0),
            duration_seconds=round(time.perf_counter() - started, 3),
            checks=[
                {"name": item.name, "passed": item.passed, "detail": item.detail}
                for item in (judgment.results if judgment else ())
            ],
            judge=judge_payload,
            metadata=dict(result.metadata or {}),
        )


def _case_checks(case: EvaluationCase) -> tuple:
    checks = list(case.checks)
    legacy = expected_output_check(case.expected_output_contains)
    if legacy is not None and all(check.name != legacy.name for check in checks):
        checks.append(legacy)
    return tuple(checks)


async def run_case(case: EvaluationCase, track: AgentTrack, run_case: CaseRunner) -> EvaluationResult:
    return await EvaluationRunner(EvaluationSuite(id="single", cases=(case,))).run_case(
        case,
        track,
        run_case,
    )

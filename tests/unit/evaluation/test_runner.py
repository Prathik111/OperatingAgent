from __future__ import annotations

import json

from common.agent import AgentRunResult, AgentTask
from common.enums import AgentTrack, RunStatus
from evaluation.runner import EvaluationRunner
from evaluation.scoring import LLMJudge
from evaluation.suite import EvaluationCase, EvaluationSuite, ui_artifact_checks

GOOD_OUTPUT = (
    "## Component Code\n```tsx\nexport function Card() {}\n```\n"
    "## Component Info\nInputs: title. States: idle. Accessibility: labelled.\n"
    "## Pacing\n1. 0ms mount\n2. 150ms fade\n"
    "## Script\n```\nHello, then it fades.\n```\n"
)
NO_SCRIPT_OUTPUT = GOOD_OUTPUT.split("## Script")[0]


def _result(output: str, status: RunStatus = RunStatus.COMPLETED) -> AgentRunResult:
    return AgentRunResult(
        status=status,
        output=output,
        duration_ms=10.0,
        llm_calls=1,
        tool_calls=0,
        total_tokens=50,
        cost=0.01,
    )


def _case(goal: str = "produce the bundle") -> EvaluationCase:
    return EvaluationCase(id="bundle", goal=goal, checks=ui_artifact_checks())


async def _run(
    case: EvaluationCase,
    output: str,
    status: RunStatus = RunStatus.COMPLETED,
    judge: LLMJudge | None = None,
):
    async def execute(task: AgentTask) -> AgentRunResult:
        return _result(output, status)

    return await EvaluationRunner(EvaluationSuite(id="s", cases=(case,))).run_case(
        case, AgentTrack.NATIVE, execute, judge=judge
    )


async def test_passing_bundle_records_check_results() -> None:
    result = await _run(_case(), GOOD_OUTPUT)
    assert result.passed
    assert [check["name"] for check in result.checks] == [
        "component_code",
        "component_info",
        "pacing",
        "script",
    ]
    assert all(check["passed"] for check in result.checks)


async def test_missing_artifact_fails_with_named_reason() -> None:
    result = await _run(_case(), NO_SCRIPT_OUTPUT)
    assert not result.passed
    failed = [check["name"] for check in result.checks if not check["passed"]]
    assert failed == ["script"]
    assert "script" in result.error


async def test_incomplete_run_fails_without_check_details() -> None:
    result = await _run(_case(), GOOD_OUTPUT, status=RunStatus.FAILED)
    assert not result.passed
    assert result.checks == []


async def test_legacy_expected_output_still_enforced() -> None:
    case = EvaluationCase(id="legacy", goal="read port", expected_output_contains="8080")
    assert (await _run(case, "port is 8080")).passed
    result = await _run(case, "port is 9090")
    assert not result.passed
    assert "expected_output_contains" in result.error


def _scripted_judge(passed: bool) -> LLMJudge:
    async def complete(messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "criteria": [
                    {"name": "component_code", "pass": passed, "reason": "scripted"},
                    {"name": "component_info", "pass": passed, "reason": "scripted"},
                    {"name": "pacing", "pass": passed, "reason": "scripted"},
                    {"name": "script", "pass": passed, "reason": "scripted"},
                ],
                "overall_comment": "scripted verdict",
            }
        )

    return LLMJudge(complete=complete, model="scripted-judge")


async def test_judge_verdict_recorded_without_touching_passed() -> None:
    for verdict_pass in (True, False):
        result = await _run(_case(), GOOD_OUTPUT, judge=_scripted_judge(verdict_pass))
        assert result.passed, "judge opinion must not flip the deterministic pass"
        assert result.judge is not None
        assert result.judge["status"] == "scored"
        assert result.judge["model"] == "scripted-judge"
        assert all(
            criterion["pass"] is verdict_pass
            for criterion in result.judge["criteria"]
        )


async def test_judge_skipped_when_run_fails_or_output_empty() -> None:
    failed = await _run(_case(), GOOD_OUTPUT, status=RunStatus.FAILED, judge=_scripted_judge(True))
    assert failed.judge is None
    empty = await _run(_case(), "", judge=_scripted_judge(True))
    assert empty.judge is None
    assert not empty.passed

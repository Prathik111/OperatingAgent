from __future__ import annotations

import pytest
from evaluation.compare import (
    compare_results,
    render_comparison_markdown,
    render_full_comparison_markdown,
)
from evaluation.runner import EvaluationResult
from evaluation.suite import ui_artifact_checks

CHECK_NAMES = [check.name for check in ui_artifact_checks()]


def _result(
    case_id: str,
    track: str,
    *,
    passed: bool = True,
    failed_checks: tuple[str, ...] = (),
    judge_score: float | None = None,
    judge_status: str = "scored",
    turns: int = 3,
    cost: float = 0.01,
) -> EvaluationResult:
    checks = [
        {"name": name, "passed": name not in failed_checks, "detail": ""}
        for name in CHECK_NAMES
    ]
    judge = None
    if judge_score is not None or judge_status == "error":
        if judge_score is not None:
            passed_count = round(judge_score * len(CHECK_NAMES))
            criteria = [
                {"name": name, "pass": index < passed_count, "reason": ""}
                for index, name in enumerate(CHECK_NAMES)
            ]
        else:
            criteria = []
        judge = {
            "model": "judge-x",
            "status": judge_status,
            "criteria": criteria,
            "overall_comment": "",
            "error": "" if judge_status == "scored" else "boom",
            "tokens": 10,
            "cost": 0.001,
        }
    return EvaluationResult(
        case_id=case_id,
        track=track,
        passed=passed,
        output="out",
        turns=turns,
        cost=cost,
        checks=checks,
        judge=judge,
    )


def test_full_markdown_keeps_deterministic_and_judge_tables_separate() -> None:
    left = [
        _result("bundle", "native", failed_checks=("script",)),
        _result("table", "native"),
    ]
    right = [
        _result("bundle", "langgraph", judge_score=0.5),
        _result("table", "langgraph", judge_status="error"),
    ]
    markdown = render_full_comparison_markdown(left, right, left_label="native", right_label="langgraph")
    assert "## Deterministic results" in markdown
    assert "## Deterministic checks" in markdown
    assert "## LLM judge" in markdown
    assert "Judge model(s): judge-x" in markdown
    assert "| Cases judged cleanly | 0 | 1 |" in markdown
    assert "| Judge errors | 0 | 1 |" in markdown
    assert "| Average judge score | — | 0.50 |" in markdown
    assert "| script | bundle | - | + |" in markdown
    assert "Deltas computed over 0 case(s)" in markdown


def test_judge_delta_only_over_cases_judged_on_both_tracks() -> None:
    left = [
        _result("a", "native", judge_score=1.0),
        _result("b", "native", judge_score=0.25),
    ]
    right = [
        _result("a", "langgraph", judge_score=0.5),
        _result("b", "langgraph", judge_status="error"),
    ]
    markdown = render_full_comparison_markdown(left, right)
    assert "| a | 1.00 | 0.50 | -0.50 |" in markdown
    assert "| b | 0.25 | — | — |" in markdown
    assert "Deltas computed over 1 case(s) judged cleanly on both tracks." in markdown


def test_compare_without_judge_renders_deterministic_only() -> None:
    left = [_result("a", "native")]
    right = [_result("a", "langgraph", passed=False)]
    markdown = render_full_comparison_markdown(left, right)
    assert "## LLM judge" not in markdown
    assert "## Deterministic results" in markdown


def test_compare_results_refuses_mismatched_case_sets() -> None:
    left = [_result("a", "native")]
    right = [_result("b", "langgraph")]
    with pytest.raises(ValueError, match="case IDs do not match"):
        compare_results(left, right)


def test_compare_results_refuses_duplicate_case_ids() -> None:
    left = [_result("a", "native"), _result("a", "native")]
    right = [_result("a", "langgraph")]
    with pytest.raises(ValueError, match="duplicate case IDs"):
        compare_results(left, right)


def test_check_matrix_includes_checks_known_only_to_one_side() -> None:
    left = [
        EvaluationResult(
            case_id="a",
            track="native",
            passed=True,
            output="o",
            checks=[{"name": "component_code", "passed": True, "detail": ""}],
        )
    ]
    right = [
        EvaluationResult(
            case_id="a",
            track="langgraph",
            passed=False,
            output="o",
            checks=[{"name": "pacing", "passed": False, "detail": "no durations"}],
        )
    ]
    markdown = render_full_comparison_markdown(left, right)
    assert "| component_code | a | + |  |" in markdown
    assert "| pacing | a |  | - |" in markdown


def test_legacy_two_column_renderer_still_works() -> None:
    left = [_result("a", "native", turns=2, cost=0.01)]
    right = [_result("a", "langgraph", turns=4, cost=0.03)]
    report = compare_results(left, right)
    markdown = render_comparison_markdown(report)
    assert "| a | pass | pass | +2 | +0.0200 |" in markdown


def test_result_roundtrip_preserves_checks_and_judge() -> None:
    from dataclasses import asdict

    original = _result("a", "native", judge_score=1.0)
    restored = EvaluationResult.from_dict(asdict(original))
    assert restored.checks == original.checks
    assert restored.judge == original.judge
    assert restored.passed is True

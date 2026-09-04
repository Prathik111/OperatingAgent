"""Comparison helpers for evaluation results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .runner import EvaluationResult


@dataclass(slots=True)
class ComparisonRow:
    case_id: str
    left_passed: bool
    right_passed: bool
    left_turns: int
    right_turns: int
    left_cost: float
    right_cost: float


@dataclass(slots=True)
class ComparisonReport:
    left_label: str
    right_label: str
    rows: list[ComparisonRow]


def compare_results(left: list[EvaluationResult], right: list[EvaluationResult]) -> ComparisonReport:
    left_ids = [result.case_id for result in left]
    right_ids = [result.case_id for result in right]
    duplicate_left = sorted(
        case_id for case_id, count in Counter(left_ids).items() if count > 1
    )
    duplicate_right = sorted(
        case_id for case_id, count in Counter(right_ids).items() if count > 1
    )
    if duplicate_left or duplicate_right:
        details = []
        if duplicate_left:
            details.append(f"left duplicate case IDs: {', '.join(duplicate_left)}")
        if duplicate_right:
            details.append(f"right duplicate case IDs: {', '.join(duplicate_right)}")
        raise ValueError("evaluation results must have unique case IDs (" + "; ".join(details) + ")")

    left_set = set(left_ids)
    right_set = set(right_ids)
    left_only = sorted(left_set - right_set)
    right_only = sorted(right_set - left_set)
    if left_only or right_only:
        details = []
        if left_only:
            details.append(f"left-only case IDs: {', '.join(left_only)}")
        if right_only:
            details.append(f"right-only case IDs: {', '.join(right_only)}")
        raise ValueError("evaluation result case IDs do not match (" + "; ".join(details) + ")")

    right_by_case = {result.case_id: result for result in right}
    rows: list[ComparisonRow] = []
    for left_result in left:
        right_result = right_by_case.get(left_result.case_id)
        if right_result is None:
            continue
        rows.append(
            ComparisonRow(
                case_id=left_result.case_id,
                left_passed=left_result.passed,
                right_passed=right_result.passed,
                left_turns=left_result.turns,
                right_turns=right_result.turns,
                left_cost=left_result.cost,
                right_cost=right_result.cost,
            )
        )
    return ComparisonReport(left_label="left", right_label="right", rows=rows)


def render_comparison_markdown(report: ComparisonReport) -> str:
    lines = [
        "# Evaluation Comparison",
        "",
        "| Case | Left | Right | d Turns | d Cost |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.case_id} | {'pass' if row.left_passed else 'fail'} | {'pass' if row.right_passed else 'fail'} | {row.right_turns - row.left_turns:+d} | {row.right_cost - row.left_cost:+.4f} |"
        )
    return "\n".join(lines)

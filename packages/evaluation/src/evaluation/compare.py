"""Comparison helpers for evaluation results.

Two honesty rules are enforced here, mirroring the report contract:

1. **Compare like with like.** Results over different suites are refused.
2. **Keep the judge separate.** The deterministic pass rate and the LLM-judge
   scores are rendered in distinct tables; a judge score never becomes the
   pass rate. Head-to-head judge deltas are only computed over cases both
   tracks had judged cleanly, and the counts are shown so an average can never
   masquerade as complete coverage.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from .runner import EvaluationResult
from .scoring import JudgeVerdict, summarize_judgments


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


def validate_compatible_suites(left: object, right: object) -> None:
    """Refuse to compare results recorded against different suites.

    Both payloads embed a ``suite_snapshot`` (see ``suite.suite_snapshot``)
    when the harness ran them; identical snapshots compare cleanly, while a
    difference in any case's id, goal, or checks is a hard error. Payloads
    from before the snapshot was recorded carry no snapshot and are accepted
    so old result files keep comparing.
    """
    left_snapshot = left.get("suite_snapshot") if isinstance(left, dict) else None
    right_snapshot = right.get("suite_snapshot") if isinstance(right, dict) else None
    if left_snapshot is None or right_snapshot is None:
        return
    left_dump = json.dumps(left_snapshot, sort_keys=True)
    right_dump = json.dumps(right_snapshot, sort_keys=True)
    if left_dump != right_dump:
        raise ValueError(
            "evaluation results describe different suites; refusing to compare "
            "(rerun each track against the same suite snapshot)"
        )


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


def _verdict(result: EvaluationResult) -> JudgeVerdict | None:
    if not result.judge:
        return None
    return JudgeVerdict.from_dict(result.judge)


def _score_of(verdicts: dict[str, JudgeVerdict | None], case_id: str) -> float | None:
    verdict = verdicts.get(case_id)
    return verdict.score if verdict is not None else None


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _pass_rate(results: list[EvaluationResult]) -> float | None:
    return (sum(1 for result in results if result.passed) / len(results)) if results else None


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Render the legacy two-column table (deterministic results only)."""
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


def render_full_comparison_markdown(
    left: list[EvaluationResult],
    right: list[EvaluationResult],
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> str:
    """Render the side-by-side report: deterministic results, then judge scores."""
    report = compare_results(left, right)
    report.left_label = left_label
    report.right_label = right_label
    left_by_case = {result.case_id: result for result in left}
    right_by_case = {result.case_id: result for result in right}

    left_rate = _pass_rate(left)
    right_rate = _pass_rate(right)
    lines: list[str] = [
        "# Evaluation Comparison",
        "",
        "## Deterministic results",
        "",
        "| Metric | " + f"{left_label} | {right_label} |",
        "| --- | --- | --- |",
        f"| Pass rate | {_format_score(left_rate)} | {_format_score(right_rate)} |",
        f"| Passed | {sum(1 for r in left if r.passed)}/{len(left)} | {sum(1 for r in right if r.passed)}/{len(right)} |",
        f"| Total turns | {sum(r.turns for r in left)} | {sum(r.turns for r in right)} |",
        f"| Total cost (USD) | {sum(r.cost for r in left):.4f} | {sum(r.cost for r in right):.4f} |",
        "",
        "| Case | " + f"{left_label} | {right_label} | d Turns | d Cost |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.case_id} | {'pass' if row.left_passed else 'fail'} | {'pass' if row.right_passed else 'fail'} | {row.right_turns - row.left_turns:+d} | {row.right_cost - row.left_cost:+.4f} |"
        )

    lines += _render_check_matrix(left_by_case, right_by_case, left_label, right_label)
    lines += _render_judge_section(left_by_case, right_by_case, left_label, right_label)
    return "\n".join(lines)


def _render_check_matrix(
    left_by_case: dict[str, EvaluationResult],
    right_by_case: dict[str, EvaluationResult],
    left_label: str,
    right_label: str,
) -> list[str]:
    check_names: list[str] = []
    for results_by_case in (left_by_case, right_by_case):
        for case_id in sorted(results_by_case):
            for check in results_by_case[case_id].checks:
                if check["name"] not in check_names:
                    check_names.append(check["name"])
    if not check_names:
        return []
    lines = [
        "",
        "## Deterministic checks",
        "",
        "Per-case named checks; `+` passed, `-` failed, blank = check not present for that case.",
        "",
        "| Check | Case | " + f"{left_label} | {right_label} |",
        "| --- | --- | --- | --- |",
    ]

    def outcome(result: EvaluationResult | None, name: str) -> str:
        if result is None:
            return ""
        for check in result.checks:
            if check["name"] == name:
                return "+" if check["passed"] else "-"
        return ""

    for name in check_names:
        for case_id in sorted(left_by_case):
            left_mark = outcome(left_by_case.get(case_id), name)
            right_mark = outcome(right_by_case.get(case_id), name)
            if not left_mark and not right_mark:
                continue
            lines.append(f"| {name} | {case_id} | {left_mark} | {right_mark} |")
    return lines


def _render_judge_section(
    left_by_case: dict[str, EvaluationResult],
    right_by_case: dict[str, EvaluationResult],
    left_label: str,
    right_label: str,
) -> list[str]:
    left_verdicts = {case_id: _verdict(result) for case_id, result in left_by_case.items()}
    right_verdicts = {case_id: _verdict(result) for case_id, result in right_by_case.items()}
    has_any = any(v is not None for v in left_verdicts.values()) or any(
        v is not None for v in right_verdicts.values()
    )
    if not has_any:
        return []
    left_summary = summarize_judgments(left_verdicts)
    right_summary = summarize_judgments(right_verdicts)
    models = sorted(
        {
            verdict.model
            for verdict in (*left_verdicts.values(), *right_verdicts.values())
            if verdict is not None and verdict.model
        }
    )
    lines = [
        "",
        "## LLM judge",
        "",
        "Non-deterministic scorer, reported separately from the deterministic pass rate.",
    ]
    if models:
        lines.append(f"Judge model(s): {', '.join(models)}.")
    lines += [
        "",
        "| Metric | " + f"{left_label} | {right_label} |",
        "| --- | --- | --- |",
        f"| Cases judged cleanly | {left_summary['judged']} | {right_summary['judged']} |",
        f"| Judge errors | {left_summary['errors']} | {right_summary['errors']} |",
        f"| Average judge score | {_format_score(left_summary['average_score'])} | {_format_score(right_summary['average_score'])} |",
        f"| Judge tokens | {left_summary['tokens']} | {right_summary['tokens']} |",
        f"| Judge cost (USD) | {left_summary['cost']:.4f} | {right_summary['cost']:.4f} |",
        "",
        "### Judge score by criterion",
        "",
        "| Criterion | " + f"{left_label} | {right_label} |",
        "| --- | --- | --- |",
    ]
    criteria = sorted(set(left_summary["criteria_pass_rate"]) | set(right_summary["criteria_pass_rate"]))
    for name in criteria:
        left_rate = left_summary["criteria_pass_rate"].get(name)
        right_rate = right_summary["criteria_pass_rate"].get(name)
        lines.append(
            f"| {name} | {_format_score(left_rate)} | {_format_score(right_rate)} |"
        )

    lines += [
        "",
        "### Judge score by case",
        "",
        "Only cases both tracks had judged cleanly are listed with a delta; `—` means not cleanly judged.",
        "",
        "| Case | " + f"{left_label} | {right_label} | d Score |",
        "| --- | --- | --- | --- |",
    ]
    both_judged = 0
    for case_id in sorted(left_by_case):
        left_score = _score_of(left_verdicts, case_id)
        right_score = _score_of(right_verdicts, case_id)
        if left_score is not None and right_score is not None:
            delta = f"{right_score - left_score:+.2f}"
            both_judged += 1
        else:
            delta = "—"
        lines.append(
            f"| {case_id} | {_format_score(left_score)} | {_format_score(right_score)} | {delta} |"
        )
    lines.append("")
    lines.append(
        f"Deltas computed over {both_judged} case(s) judged cleanly on both tracks."
    )
    return lines

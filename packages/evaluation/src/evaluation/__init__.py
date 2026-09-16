"""Reusable evaluation harness for OperatingAgent."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .compare import (
    compare_results,
    render_comparison_markdown,
    render_full_comparison_markdown,
    validate_compatible_suites,
)
from .execution import run_suite
from .runner import EvaluationResult, EvaluationRunner, run_case
from .scoring import JudgeVerdict, LLMJudge, judge_from_registry, summarize_judgments
from .suite import (
    EvaluationCase,
    EvaluationSuite,
    default_suite,
    load_suite,
    save_suite,
    suite_from_snapshot,
    suite_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Evaluation harness utilities for the agent tracks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_suite = sub.add_parser("suite", help="Write the built-in example suite to JSON.")
    p_suite.add_argument(
        "--out", default="evaluation-suite.json", help="Where to write the suite JSON."
    )

    p_compare = sub.add_parser(
        "compare", help="Compare two result files and render markdown."
    )
    p_compare.add_argument("left", help="Left result JSON file.")
    p_compare.add_argument("right", help="Right result JSON file.")
    p_compare.add_argument("--out", default="", help="Optional markdown output path.")

    p_run = sub.add_parser("run", help="Run the built-in suite against real tracks.")
    p_run.add_argument("--suite", default="", help="Optional suite JSON file.")
    p_run.add_argument("--out", default="evaluation-results.json", help="Where to write the results JSON.")
    p_run.add_argument(
        "--judge-model",
        default="",
        help=(
            "Optional LLM judge: a model name from the runtime registry "
            "(e.g. a Groq model). One judge instance is shared by every "
            "track; scores are recorded separately from deterministic checks."
        ),
    )

    p_judge = sub.add_parser(
        "judge",
        help="Re-judge an existing results file with an LLM, without re-running.",
    )
    p_judge.add_argument("results", help="Results JSON file to judge.")
    p_judge.add_argument("--suite", default="", help="Optional suite JSON file naming the checks (defaults to the built-in suite when the results file ran it).")
    p_judge.add_argument(
        "--judge-model",
        required=True,
        help="Model name from the runtime registry used as the judge.",
    )
    p_judge.add_argument("--out", default="", help="Optional output path (default: <results>-judged.json).")

    args = parser.parse_args()

    if args.command == "suite":
        save_suite(default_suite(), Path(args.out))
        return

    if args.command == "compare":
        left_payload = json.loads(Path(args.left).read_text(encoding="utf-8"))
        right_payload = json.loads(Path(args.right).read_text(encoding="utf-8"))
        validate_compatible_suites(left_payload, right_payload)
        left = load_results(Path(args.left))
        right = load_results(Path(args.right))
        markdown = render_full_comparison_markdown(
            left,
            right,
            left_label=Path(args.left).stem,
            right_label=Path(args.right).stem,
        )
        if args.out:
            Path(args.out).write_text(markdown, encoding="utf-8")
        else:
            print(markdown)
        return

    if args.command == "run":
        suite = load_suite(Path(args.suite)) if args.suite else default_suite()
        results = asyncio.run(run_suite(suite, judge_model=args.judge_model))
        payload = {
            "suite": suite.id,
            "suite_snapshot": suite_snapshot(suite),
            "results": [asdict(result) for result in results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if args.command == "judge":
        asyncio.run(
            _judge_results_file(
                Path(args.results),
                suite_path=Path(args.suite) if args.suite else None,
                judge_model=args.judge_model,
                out=Path(args.out) if args.out else None,
            )
        )
        return


def _resolve_judge_suite(payload: object, suite_path: Path | None):
    """Pick the checks source for re-judging.

    An explicit ``--suite`` file always wins. Otherwise, when the results file
    recorded a suite snapshot (the normal harness run), that suite is used so
    re-judging and comparing stay in sync with what actually ran. Finally, the
    built-in default suite covers legacy files from before snapshots existed.
    """
    from .suite import default_suite, load_suite, suite_from_snapshot

    if suite_path is not None:
        return load_suite(suite_path)
    if isinstance(payload, dict) and payload.get("suite_snapshot") is not None:
        return suite_from_snapshot(payload["suite_snapshot"])
    suite_id = payload.get("suite") if isinstance(payload, dict) else None
    if suite_id == "default":
        return default_suite()
    return None


async def _judge_results_file(
    results_path: Path,
    *,
    suite_path: Path | None,
    judge_model: str,
    out: Path | None,
) -> None:
    """Score an existing results file's outputs with the LLM judge.

    The pass/fail columns in the file are untouched: judge verdicts are added
    beside them, so the deterministic record stays exactly what it was.
    """
    from .execution import open_evaluation_environment
    from .runner import _case_checks

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    suite = _resolve_judge_suite(payload, suite_path)
    checks_by_case = {case.id: _case_checks(case) for case in suite.cases} if suite else {}
    goals_by_case = {case.id: case.goal for case in suite.cases} if suite else {}

    async with open_evaluation_environment() as environment:
        registry = getattr(environment.native_runtime, "models", None)
        if registry is None:
            raise RuntimeError("LLM judge requested but no model registry is available")
        judge = judge_from_registry(registry, judge_model)
        for row in rows:
            case_id = str(row.get("case_id", ""))
            checks = checks_by_case.get(case_id, ())
            goal = goals_by_case.get(
                case_id, str((row.get("metadata") or {}).get("goal", ""))
            )
            if not checks:
                row["judge"] = JudgeVerdict(
                    model=judge.model, status="error", error="no checks known for this case"
                ).to_dict()
                continue
            verdict = await judge.judge(goal=goal, output=str(row.get("output", "")), checks=checks)
            row["judge"] = verdict.to_dict()

    destination = out or results_path.with_name(results_path.stem + "-judged.json")
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"judged results written to {destination}")


def load_results(path: Path) -> list[EvaluationResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    return [EvaluationResult.from_dict(item) for item in payload]


__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "EvaluationRunner",
    "EvaluationSuite",
    "JudgeVerdict",
    "LLMJudge",
    "compare_results",
    "default_suite",
    "judge_from_registry",
    "load_suite",
    "main",
    "render_comparison_markdown",
    "render_full_comparison_markdown",
    "run_case",
    "run_suite",
    "save_suite",
    "suite_from_snapshot",
    "suite_snapshot",
    "summarize_judgments",
    "validate_compatible_suites",
]

"""Reusable evaluation harness for OperatingAgent."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .compare import compare_results, render_comparison_markdown
from .execution import run_suite
from .runner import EvaluationResult, EvaluationRunner, run_case
from .suite import (
    EvaluationCase,
    EvaluationSuite,
    default_suite,
    load_suite,
    save_suite,
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

    args = parser.parse_args()

    if args.command == "suite":
        save_suite(default_suite(), Path(args.out))
        return

    if args.command == "compare":
        left = load_results(Path(args.left))
        right = load_results(Path(args.right))
        markdown = render_comparison_markdown(compare_results(left, right))
        if args.out:
            Path(args.out).write_text(markdown, encoding="utf-8")
        else:
            print(markdown)
        return

    if args.command == "run":
        suite = load_suite(Path(args.suite)) if args.suite else default_suite()
        results = asyncio.run(run_suite(suite))
        payload = {
            "suite": suite.id,
            "results": [asdict(result) for result in results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return


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
    "compare_results",
    "default_suite",
    "load_suite",
    "main",
    "render_comparison_markdown",
    "run_case",
    "run_suite",
    "save_suite",
]

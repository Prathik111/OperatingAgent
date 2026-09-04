"""Suite and case definitions for benchmarkable agent runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class EvaluationCase:
    id: str
    goal: str
    working_directory: str = "."
    expected_output_contains: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EvaluationSuite:
    id: str
    cases: tuple[EvaluationCase, ...]


def default_suite() -> EvaluationSuite:
    return EvaluationSuite(
        id="default",
        cases=(
            EvaluationCase(
                id="hello",
                goal="Reply with a short friendly greeting.",
            ),
            EvaluationCase(
                id="status",
                goal="Summarize what you can do in one sentence.",
            ),
        ),
    )


def load_suite(path: Path) -> EvaluationSuite:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(
        EvaluationCase(
            id=str(item["id"]),
            goal=str(item["goal"]),
            working_directory=str(item.get("working_directory", ".")),
            expected_output_contains=str(item.get("expected_output_contains", "")),
            metadata=dict(item.get("metadata", {}) or {}),
        )
        for item in payload.get("cases", [])
    )
    return EvaluationSuite(id=str(payload.get("id", path.stem)), cases=cases)


def save_suite(suite: EvaluationSuite, path: Path) -> None:
    payload = {
        "id": suite.id,
        "cases": [
            {
                "id": case.id,
                "goal": case.goal,
                "working_directory": case.working_directory,
                "expected_output_contains": case.expected_output_contains,
                "metadata": case.metadata,
            }
            for case in suite.cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

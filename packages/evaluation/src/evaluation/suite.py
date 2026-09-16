"""Suite and case definitions for benchmarkable agent runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.judging import OutputCheck


@dataclass(slots=True, frozen=True)
class EvaluationCase:
    id: str
    goal: str
    working_directory: str = "."
    expected_output_contains: str = ""
    checks: tuple[OutputCheck, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EvaluationSuite:
    id: str
    cases: tuple[EvaluationCase, ...]


def ui_artifact_checks(
    *,
    info_terms: Sequence[str] = ("input", "state", "accessib"),
    include_code: bool = True,
    include_info: bool = True,
    include_pacing: bool = True,
    include_script: bool = True,
) -> tuple[OutputCheck, ...]:
    """The standard artifact checks for a generated UI bundle.

    The component itself is deliberately framework-free: any fenced code block
    counts as the implementation, in any language. The remaining checks pin the
    artifacts that must accompany the code — the component's documentation, the
    timing plan, and the narration script — each anchored to a labeled section
    so the judge can tell *which* artifact is missing. The pacing check
    requires an actual duration (``\\d+\\s*ms``), not just the letters "ms",
    which ordinary words like "items" would otherwise satisfy.
    """
    checks: list[OutputCheck] = []
    if include_code:
        checks.append(
            OutputCheck(name="component_code", kind="labeled_code_block", labels=("component code",))
        )
    if include_info:
        checks.append(
            OutputCheck(
                name="component_info",
                kind="labeled_contains",
                labels=("component info",),
                terms=tuple(info_terms),
            )
        )
    if include_pacing:
        checks.append(
            OutputCheck(
                name="pacing",
                kind="labeled_regex",
                labels=("pacing",),
                pattern=r"\d+\s*ms",
            )
        )
    if include_script:
        checks.append(
            OutputCheck(name="script", kind="labeled_code_block", labels=("script",))
        )
    return tuple(checks)


def default_suite() -> EvaluationSuite:
    return EvaluationSuite(
        id="default",
        cases=(
            EvaluationCase(
                id="ui_component_bundle",
                goal=(
                    "Produce a complete interactive-card UI bundle. Structure the answer "
                    "with these labeled sections, each introduced by its exact name: "
                    "\"Component Code\" - the implementation in a fenced code block, in any "
                    "framework or plain HTML; \"Component Info\" - document the inputs, the "
                    "states, and the accessibility behaviour; \"Pacing\" - a numbered reveal "
                    "timeline with every duration written in ms; \"Script\" - the narration "
                    "script inside a fenced code block."
                ),
                checks=ui_artifact_checks(),
            ),
            EvaluationCase(
                id="ui_component_framework_free",
                goal=(
                    "Generate a data-table UI component in any framework of your choice. "
                    "Include a \"Component Code\" section with the implementation in a fenced "
                    "code block, and a \"Component Info\" section that documents the inputs "
                    "and the states."
                ),
                checks=ui_artifact_checks(include_pacing=False, include_script=False),
            ),
            EvaluationCase(
                id="ui_component_documented",
                goal=(
                    "Build a settings-panel UI component and document it fully. Include a "
                    "\"Component Code\" section with the implementation in a fenced code block "
                    "(any framework), and a \"Component Info\" section covering inputs, states, "
                    "styling, events, and accessibility."
                ),
                checks=ui_artifact_checks(
                    info_terms=("input", "state", "style", "event", "accessib"),
                    include_pacing=False,
                    include_script=False,
                ),
            ),
            EvaluationCase(
                id="reveal_pacing_and_script",
                goal=(
                    "Plan the presentation layer for a three-step onboarding reveal. Provide a "
                    "\"Pacing\" section with a numbered timeline where every duration is written "
                    "in ms, and a \"Script\" section with the narration text inside a fenced "
                    "code block."
                ),
                checks=ui_artifact_checks(include_code=False, include_info=False),
            ),
        ),
    )


def suite_snapshot(suite: EvaluationSuite) -> dict[str, Any]:
    """A JSON-safe snapshot of the suite, enough to re-judge or verify later."""
    return {
        "id": suite.id,
        "cases": [
            {
                "id": case.id,
                "goal": case.goal,
                "working_directory": case.working_directory,
                "expected_output_contains": case.expected_output_contains,
                "checks": [check.to_dict() for check in case.checks],
                "metadata": case.metadata,
            }
            for case in suite.cases
        ],
    }


def suite_from_snapshot(payload: object) -> EvaluationSuite:
    """Rebuild a suite from the JSON produced by :func:`suite_snapshot`."""
    if not isinstance(payload, dict):
        raise TypeError("suite snapshot must be an object")
    cases = tuple(
        EvaluationCase(
            id=str(item["id"]),
            goal=str(item["goal"]),
            working_directory=str(item.get("working_directory", ".")),
            expected_output_contains=str(item.get("expected_output_contains", "")),
            checks=tuple(
                OutputCheck.from_dict(check) for check in (item.get("checks") or [])
            ),
            metadata=dict(item.get("metadata", {}) or {}),
        )
        for item in payload.get("cases", [])
    )
    return EvaluationSuite(id=str(payload.get("id", "")), cases=cases)


def load_suite(path: Path) -> EvaluationSuite:
    suite = suite_from_snapshot(json.loads(path.read_text(encoding="utf-8")))
    if not suite.id:
        return EvaluationSuite(id=path.stem, cases=suite.cases)
    return suite


def save_suite(suite: EvaluationSuite, path: Path) -> None:
    path.write_text(json.dumps(suite_snapshot(suite), indent=2), encoding="utf-8")

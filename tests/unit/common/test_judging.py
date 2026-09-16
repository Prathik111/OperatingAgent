from __future__ import annotations

import pytest
from common.judging import (
    CheckResult,
    Judgment,
    OutputCheck,
    expected_output_check,
    judge_output,
)


def _bundle_output() -> str:
    return (
        "## Component Code\n"
        "```tsx\n"
        "export function Card() { return <div />; }\n"
        "```\n"
        "## Component Info\n"
        "Inputs: title, onDismiss. States: idle, busy. Accessibility: labelled by title.\n"
        "## Pacing\n"
        "1. 0ms card mounts\n"
        "2. 200ms title fades in\n"
        "## Script\n"
        "```\n"
        "First the card appears, then the title settles.\n"
        "```\n"
    )


def test_labeled_code_block_accepts_any_language() -> None:
    checks = (OutputCheck(name="component_code", kind="labeled_code_block", labels=("component code",)),)
    for language in ("tsx", "python", "", "vue"):
        output = f"Component Code\n```{language}\nrender()\n```"
        assert judge_output(output, checks).passed, language


def test_labeled_code_block_requires_label_then_fence() -> None:
    check = OutputCheck(name="component_code", kind="labeled_code_block", labels=("component code",))
    assert not judge_output("```tsx\nrender()\n```", (check,)).passed
    assert not judge_output("Component Code\nno fence here", (check,)).passed


def test_labeled_code_block_can_require_in_block_terms() -> None:
    check = OutputCheck(
        name="component_code",
        kind="labeled_code_block",
        labels=("component code",),
        terms=("export function",),
    )
    assert judge_output("Component Code\n```tsx\nexport function Card() {}\n```", (check,)).passed
    assert not judge_output("Component Code\n```tsx\nconst x = 1\n```", (check,)).passed


def test_section_boundary_ignores_capitalized_prose_inside_fenced_artifact() -> None:
    check = OutputCheck(
        name="component_code",
        kind="labeled_code_block",
        labels=("component code",),
        terms=("Component",),
    )
    output = "Component Code\n```tsx\nComponent\nreturn <Card />\n```\nComponent Info\ninputs: title"
    assert judge_output(output, (check,)).passed


def test_section_boundary_does_not_cut_capitalized_body_text() -> None:
    check = OutputCheck(
        name="info",
        kind="labeled_contains",
        labels=("component info",),
        terms=("Inputs", "States"),
    )
    output = "Component Info\nInputs: title. States: idle.\nPacing\n200ms"
    assert judge_output(output, (check,)).passed


def test_labeled_contains_matches_terms_after_label_case_insensitively() -> None:
    check = OutputCheck(name="pacing", kind="labeled_contains", labels=("pacing",), terms=("ms",))
    assert judge_output("PACING\nstep 1: 200ms", (check,)).passed
    assert not judge_output("PACING\nstep 1: 0.2 seconds", (check,)).passed
    assert not judge_output("OTHER\n200ms", (check,)).passed


def test_labeled_regex_requires_real_duration_not_embedded_ms() -> None:
    check = OutputCheck(
        name="pacing", kind="labeled_regex", labels=("pacing",), pattern=r"\d+\s*ms"
    )
    assert judge_output("Pacing\n1. 200ms fade in\n2. 400 ms settle", (check,)).passed
    # "ms" inside ordinary words must not count as a duration.
    assert not judge_output("Pacing\nThe reveal follows the items in order", (check,)).passed
    assert not judge_output("200ms everywhere but no label", (check,)).passed
    assert not judge_output("Pacing\nunfolds over time", (check,)).passed


def test_labeled_regex_reports_invalid_pattern() -> None:
    check = OutputCheck(name="bad", kind="labeled_regex", labels=("l",), pattern="([")
    result = judge_output("l\nanything", (check,)).results[0]
    assert not result.passed
    assert "invalid pattern" in result.detail


def test_any_contains_passes_on_first_matching_term() -> None:
    check = OutputCheck(name="info", kind="any_contains", terms=("inputs", "props"))
    assert judge_output("documents the props", (check,)).passed
    assert not judge_output("documents nothing", (check,)).passed


def test_regex_check() -> None:
    check = OutputCheck(name="timeline", kind="regex", pattern=r"\d+\s*ms")
    assert judge_output("wait 250 ms then settle", (check,)).passed
    assert not judge_output("wait a moment", (check,)).passed


def test_empty_output_fails_every_check_with_reason() -> None:
    checks = (
        OutputCheck(name="component_code", kind="labeled_code_block", labels=("component code",)),
        OutputCheck(name="pacing", kind="labeled_contains", labels=("pacing",), terms=("ms",)),
    )
    judgment = judge_output("   \n", checks)
    assert not judgment.passed
    assert [result.detail for result in judgment.results] == ["empty output", "empty output"]


def test_full_bundle_passes_all_artifact_checks() -> None:
    from evaluation.suite import ui_artifact_checks

    judgment = judge_output(_bundle_output(), ui_artifact_checks())
    assert judgment.passed, judgment.failure_summary()
    assert [result.name for result in judgment.results] == [
        "component_code",
        "component_info",
        "pacing",
        "script",
    ]


def test_missing_script_artifact_is_reported_by_name() -> None:
    from evaluation.suite import ui_artifact_checks

    output = _bundle_output().split("## Script")[0]
    judgment = judge_output(output, ui_artifact_checks())
    assert not judgment.passed
    assert judgment.failed[0].name == "script"


def test_pacing_section_without_durations_fails_default_check() -> None:
    from evaluation.suite import ui_artifact_checks

    output = _bundle_output().replace(
        "1. 0ms card mounts\n2. 200ms title fades in",
        "The reveal follows the items in a calm order",
    )
    judgment = judge_output(output, ui_artifact_checks())
    failed = [result.name for result in judgment.failed]
    assert failed == ["pacing"]


def test_judgment_with_no_checks_passes_on_nonempty_output() -> None:
    assert judge_output("anything", ()).passed


def test_expected_output_check_wraps_legacy_field() -> None:
    assert expected_output_check("") is None
    check = expected_output_check("Port 8080")
    assert check is not None
    assert check.name == "expected_output_contains"
    assert judge_output("the configured Port 8080 is open", (check,)).passed
    assert not judge_output("the port is 9090", (check,)).passed


def test_output_check_roundtrip_and_validation() -> None:
    check = OutputCheck(
        name="component_code",
        kind="labeled_code_block",
        labels=("component code",),
        terms=("export",),
        languages=("tsx", "jsx"),
        min_blocks=2,
    )
    restored = OutputCheck.from_dict(check.to_dict())
    assert restored == check
    with pytest.raises(ValueError, match="unknown check kind"):
        OutputCheck.from_dict({"name": "x", "kind": "vibes"})
    with pytest.raises(ValueError, match="requires a name"):
        OutputCheck.from_dict({"name": " ", "kind": "regex", "pattern": "x"})
    with pytest.raises(ValueError, match="requires a pattern"):
        OutputCheck.from_dict({"name": "x", "kind": "regex"})
    with pytest.raises(ValueError, match="requires labels"):
        OutputCheck.from_dict({"name": "x", "kind": "labeled_code_block"})
    with pytest.raises(ValueError, match="requires terms"):
        OutputCheck.from_dict({"name": "x", "kind": "contains"})
    with pytest.raises(ValueError, match="requires a pattern"):
        OutputCheck.from_dict({"name": "x", "kind": "labeled_regex", "labels": ["l"]})
    with pytest.raises(ValueError, match="requires labels"):
        OutputCheck.from_dict({"name": "x", "kind": "labeled_regex", "pattern": "x"})


def test_failure_summary_lists_only_failures() -> None:
    judgment = Judgment(
        passed=False,
        results=(
            CheckResult(name="a", passed=True, detail="ok"),
            CheckResult(name="b", passed=False, detail="missing terms: z"),
        ),
    )
    assert judgment.failure_summary() == "b: missing terms: z"

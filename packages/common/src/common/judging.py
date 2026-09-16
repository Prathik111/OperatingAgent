"""Deterministic output judging shared by the evaluation harness and the API.

A judgement is a set of named checks over a finished run's output. Checks are
deliberately structural, not semantic: they reproduce byte-for-byte, they stay
framework-agnostic (a code artifact is accepted in any language), and every
failure carries a human-readable reason so a report can show *which* artifact
was missing rather than a bare pass/fail.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

FENCED_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

CHECK_KINDS = (
    "contains",
    "any_contains",
    "regex",
    "labeled_code_block",
    "labeled_contains",
    "labeled_regex",
)


@dataclass(slots=True, frozen=True)
class OutputCheck:
    """One named, deterministic check over an agent's final output."""

    name: str
    kind: str
    terms: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    pattern: str = ""
    languages: tuple[str, ...] = ()
    min_blocks: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.terms:
            payload["terms"] = list(self.terms)
        if self.labels:
            payload["labels"] = list(self.labels)
        if self.pattern:
            payload["pattern"] = self.pattern
        if self.languages:
            payload["languages"] = list(self.languages)
        if self.min_blocks != 1:
            payload["min_blocks"] = self.min_blocks
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputCheck:
        kind = str(data.get("kind", "") or "")
        if kind not in CHECK_KINDS:
            raise ValueError(
                f"unknown check kind {kind!r} (expected one of: {', '.join(CHECK_KINDS)})"
            )
        name = str(data.get("name", "") or "").strip()
        if not name:
            raise ValueError("output check requires a name")
        terms = tuple(str(term) for term in (data.get("terms") or ()) if str(term).strip())
        labels = tuple(str(label) for label in (data.get("labels") or ()) if str(label).strip())
        pattern = str(data.get("pattern", "") or "")
        languages = tuple(
            str(language).strip().lower()
            for language in (data.get("languages") or ())
            if str(language).strip()
        )
        try:
            min_blocks = int(data.get("min_blocks", 1) or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_blocks must be an integer: {exc}") from exc
        if min_blocks < 1:
            raise ValueError("min_blocks must be >= 1")
        if kind in ("contains", "any_contains") and not terms:
            raise ValueError(f"check {name!r} of kind {kind!r} requires terms")
        if kind in ("regex", "labeled_regex") and not pattern:
            raise ValueError(f"check {name!r} of kind {kind!r} requires a pattern")
        if kind in ("labeled_code_block", "labeled_contains", "labeled_regex") and not labels:
            raise ValueError(f"check {name!r} of kind {kind!r} requires labels")
        return cls(
            name=name,
            kind=kind,
            terms=terms,
            labels=labels,
            pattern=pattern,
            languages=languages,
            min_blocks=min_blocks,
        )


@dataclass(slots=True, frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True, frozen=True)
class Judgment:
    passed: bool
    results: tuple[CheckResult, ...]

    @property
    def failed(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def failure_summary(self) -> str:
        if self.passed or not self.results:
            return ""
        return "; ".join(f"{result.name}: {result.detail}" for result in self.failed)


def expected_output_check(expected: str) -> OutputCheck | None:
    """Wrap the legacy ``expected_output_contains`` field as a named check."""
    cleaned = (expected or "").strip()
    if not cleaned:
        return None
    return OutputCheck(name="expected_output_contains", kind="contains", terms=(cleaned,))


def judge_output(output: str, checks: Sequence[OutputCheck]) -> Judgment:
    """Judge a finished run's output against an ordered set of named checks."""
    text = output or ""
    if not text.strip():
        return Judgment(
            passed=False,
            results=tuple(
                CheckResult(name=check.name, passed=False, detail="empty output")
                for check in checks
            ),
        )
    if not checks:
        return Judgment(passed=True, results=())
    results = tuple(_run_check(check, text) for check in checks)
    return Judgment(passed=all(result.passed for result in results), results=results)


def _run_check(check: OutputCheck, output: str) -> CheckResult:
    if check.kind == "contains":
        missing = [term for term in check.terms if term.lower() not in output.lower()]
        if missing:
            return CheckResult(check.name, False, "missing terms: " + ", ".join(missing))
        return CheckResult(check.name, True, "all terms found")
    if check.kind == "any_contains":
        found = next((term for term in check.terms if term.lower() in output.lower()), None)
        if found is None:
            return CheckResult(
                check.name, False, "none of the terms found: " + ", ".join(check.terms)
            )
        return CheckResult(check.name, True, f"found term: {found}")
    if check.kind == "regex":
        try:
            matched = re.search(check.pattern, output) is not None
        except re.error as exc:
            return CheckResult(check.name, False, f"invalid pattern: {exc}")
        if not matched:
            return CheckResult(check.name, False, f"pattern not found: {check.pattern}")
        return CheckResult(check.name, True, "pattern found")
    if check.kind == "labeled_code_block":
        return _judge_labeled_code_block(check, output)
    if check.kind == "labeled_contains":
        return _judge_labeled_contains(check, output)
    return _judge_labeled_regex(check, output)


def _find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    positions: list[int] = []
    index = haystack.find(needle)
    while index != -1:
        positions.append(index)
        index = haystack.find(needle, index + len(needle))
    return positions


def _label_spans(check: OutputCheck, output: str) -> list[tuple[int, int]]:
    lowered = output.lower()
    spans = [
        (position, len(label))
        for label in check.labels
        for position in _find_all(lowered, label.lower())
    ]
    return sorted(spans)


_SECTION_HEADING = re.compile(
    # Markdown headings are unambiguous.  For plain-text headings, require
    # title-case words (or an all-caps label) so normal prose such as
    # ``The reveal follows...`` is not treated as the next section.
    r"(?im)^\s*(?:#{1,6}\s+.+|(?:[A-Z][A-Za-z0-9_/-]*)(?:\s+[A-Z][A-Za-z0-9_/-]*){0,8}:?\s*)$"
)


def _section_after_label(output: str, position: int, length: int) -> str:
    """Limit evidence to the artifact section following a label."""
    start = position + length
    section = output[start:]
    offset = 0
    in_fence = False
    for line in section.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and _SECTION_HEADING.match(line):
            return section[:offset]
        offset += len(line)
    return section


def _judge_labeled_code_block(check: OutputCheck, output: str) -> CheckResult:
    spans = _label_spans(check, output)
    if not spans:
        return CheckResult(
            check.name, False, "no label found: " + ", ".join(check.labels)
        )
    for position, length in spans:
        after = _section_after_label(output, position, length)
        blocks = list(FENCED_BLOCK.finditer(after))
        if len(blocks) < check.min_blocks:
            continue
        for match in blocks:
            language = match.group(1).strip().lower()
            if check.languages and language not in check.languages:
                continue
            body = match.group(2)
            missing = [term for term in check.terms if term.lower() not in body.lower()]
            if missing:
                continue
            detail = "code block found"
            if check.languages:
                detail += f" (language: {language or 'none'})"
            return CheckResult(check.name, True, detail)
    reasons = ["no fenced code block with the required properties after any label"]
    if check.languages:
        reasons.append("accepted languages: " + ", ".join(check.languages))
    if check.terms:
        reasons.append("required in-block terms: " + ", ".join(check.terms))
    return CheckResult(check.name, False, "; ".join(reasons))


def _judge_labeled_contains(check: OutputCheck, output: str) -> CheckResult:
    spans = _label_spans(check, output)
    if not spans:
        return CheckResult(
            check.name, False, "no label found: " + ", ".join(check.labels)
        )
    for position, length in spans:
        section = _section_after_label(output, position, length).lower()
        missing = [term for term in check.terms if term.lower() not in section]
        if not missing:
            return CheckResult(check.name, True, "all terms found after label")
    return CheckResult(
        check.name,
        False,
        "terms not found after any label: " + ", ".join(check.terms),
    )


def _judge_labeled_regex(check: OutputCheck, output: str) -> CheckResult:
    spans = _label_spans(check, output)
    if not spans:
        return CheckResult(
            check.name, False, "no label found: " + ", ".join(check.labels)
        )
    try:
        pattern = re.compile(check.pattern)
    except re.error as exc:
        return CheckResult(check.name, False, f"invalid pattern: {exc}")
    for position, length in spans:
        if pattern.search(_section_after_label(output, position, length)):
            return CheckResult(check.name, True, "pattern found after label")
    return CheckResult(
        check.name,
        False,
        f"pattern not found after any label: {check.pattern}",
    )

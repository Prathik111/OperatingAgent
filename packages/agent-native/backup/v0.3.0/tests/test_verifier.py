"""Verifier tests - objective checks + explicit UNVERIFIABLE (decision #2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_native.types import PlanStep, StepKind, ToolCallResult
from agent_native.verifier import VerificationOutcome, Verifier


def step(kind: StepKind, check: str | None, tool: str = "run_command") -> PlanStep:
    return PlanStep(id="s1", description="d", kind=kind, tool_name=tool, check=check)


def ok(output=None) -> ToolCallResult:
    return ToolCallResult(success=True, output=output if output is not None else "0")


def fail() -> ToolCallResult:
    return ToolCallResult(success=False, output=None, error="boom")


def test_analysis_step_never_passes_silently(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    assert v.verify(step(StepKind.ANALYSIS, None), ok()) == VerificationOutcome.UNVERIFIABLE


def test_tool_step_without_check_is_unverifiable(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    assert v.verify(step(StepKind.TOOL, None), ok()) == VerificationOutcome.UNVERIFIABLE


def test_unknown_check_descriptor_is_unverifiable(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    assert v.verify(step(StepKind.TOOL, "build_passed"), ok()) == VerificationOutcome.UNVERIFIABLE


def test_exit_code_success(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    assert v.verify(step(StepKind.TOOL, "exit_code=0"), ok("0")) == VerificationOutcome.PASS
    assert v.verify(step(StepKind.TOOL, "exit_code=0"), ok()) == VerificationOutcome.PASS


def test_exit_code_failure(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    assert v.verify(step(StepKind.TOOL, "exit_code=0"), fail()) == VerificationOutcome.FAIL
    assert v.verify(step(StepKind.TOOL, "exit_code=0"), ok("1")) == VerificationOutcome.FAIL


def test_file_exists(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    target = tmp_path / "out.txt"
    assert v.verify(step(StepKind.TOOL, "file_exists=out.txt"), ok()) == VerificationOutcome.FAIL
    target.write_text("x")
    assert v.verify(step(StepKind.TOOL, "file_exists=out.txt"), ok()) == VerificationOutcome.PASS
    assert v.verify(step(StepKind.TOOL, "file_exists=missing.txt"), ok()) == VerificationOutcome.FAIL


def test_file_absent_and_dir_exists(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    (tmp_path / "keep").mkdir()
    assert v.verify(step(StepKind.TOOL, "file_absent=temp.log"), ok()) == VerificationOutcome.PASS
    assert v.verify(step(StepKind.TOOL, "dir_exists=keep"), ok()) == VerificationOutcome.PASS
    assert v.verify(step(StepKind.TOOL, "dir_exists=nope"), ok()) == VerificationOutcome.FAIL


def test_absolute_paths_resolved_absolutely(tmp_path: Path):
    v = Verifier(workspace=tmp_path)
    target = tmp_path / "abs.txt"
    target.write_text("x")
    assert v.verify(step(StepKind.TOOL, f"file_exists={target}"), ok()) == VerificationOutcome.PASS
"""Verifier - objective checks for plan steps.

Decision #2: steps the planner cannot attach an objective check to are marked
kind=ANALYSIS. For those, verify() returns UNVERIFIABLE - it never silently
returns True. The executor treats UNVERIFIABLE as "passes by design" but
emits STEP_UNVERIFIABLE so the run record shows the step was not checked.

TOOL steps carry `check` descriptors understood here:
  exit_code=0           tool result success and exit code semantics
  file_exists=<path>    path exists (resolved against the workspace/cwd)
  file_absent=<path>    path does not exist
  dir_exists=<path>     directory exists
Any other descriptor -> UNVERIFIABLE (again: no silent pass).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from ..types import PlanStep, StepKind, ToolCallResult


class VerificationOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"


class Verifier:
    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = workspace or Path.cwd()

    def verify(self, step: PlanStep, result: ToolCallResult) -> VerificationOutcome:
        """Objective verification. Never returns PASS for an ANALYSIS step."""
        if step.kind == StepKind.ANALYSIS:
            return VerificationOutcome.UNVERIFIABLE

        if step.check is None:
            # A TOOL step without a check descriptor is a planner bug - fail
            # loudly rather than passing silently.
            return VerificationOutcome.UNVERIFIABLE

        check = step.check.strip()
        if check.startswith("exit_code="):
            wanted = check.split("=", 1)[1].strip()
            if not result.success:
                return VerificationOutcome.FAIL
            if wanted and result.output is not None:
                try:
                    return (
                        VerificationOutcome.PASS
                        if str(result.output).strip() == wanted
                        else VerificationOutcome.FAIL
                    )
                except (TypeError, ValueError):
                    return VerificationOutcome.FAIL
            return VerificationOutcome.PASS

        if check.startswith("file_exists="):
            path = self._resolve(check.split("=", 1)[1].strip())
            return VerificationOutcome.PASS if path.exists() else VerificationOutcome.FAIL

        if check.startswith("file_absent="):
            path = self._resolve(check.split("=", 1)[1].strip())
            return VerificationOutcome.PASS if not path.exists() else VerificationOutcome.FAIL

        if check.startswith("dir_exists="):
            path = self._resolve(check.split("=", 1)[1].strip())
            return VerificationOutcome.PASS if path.is_dir() else VerificationOutcome.FAIL

        return VerificationOutcome.UNVERIFIABLE

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        return self.workspace / p

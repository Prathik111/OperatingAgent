"""Scorers: deterministic checks plus an optional LLM judge.

The judge implementation lives in ``common.llm_judging`` so the API service
and this harness share one implementation (and therefore one fairness
contract). This module re-exports it and keeps the provider-registry adapter
that belongs with the harness.

Fairness rules, enforced structurally in ``common.llm_judging``:

1. **One judge for every track** — a single judge instance (one model, one
   prompt, temperature 0) is built once per run and reused for every track.
2. **Blind judging** — the prompt contains only the goal, the output, and the
   rubric. It never contains the track, case id, or run metadata.
3. **Separate reporting** — the judge score never overwrites the deterministic
   pass/fail; it is stored beside it and rendered in its own tables.
4. **Fail-visible** — a judge that errors, returns unparseable JSON, or does
   not score exactly the requested criteria records ``status="error"`` and a
   ``None`` score; errored verdicts are counted, never averaged in as passes.
5. **Judge overhead accounted separately** — judging tokens and cost are
   recorded on the verdict, not on the judged run.
"""

from __future__ import annotations

from common.llm_judging import (
    DEFAULT_RUBRIC,
    GENERIC_CRITERION,
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    Complete,
    JudgeCriterion,
    JudgeCriterionVerdict,
    JudgeSummary,
    JudgeVerdict,
    LLMJudge,
    judge_from_registry,
    summarize_judgments,
)

__all__ = [
    "DEFAULT_RUBRIC",
    "GENERIC_CRITERION",
    "JUDGE_PROMPT_TEMPLATE",
    "JUDGE_SYSTEM_PROMPT",
    "Complete",
    "JudgeCriterion",
    "JudgeCriterionVerdict",
    "JudgeSummary",
    "JudgeVerdict",
    "LLMJudge",
    "judge_from_registry",
    "summarize_judgments",
]

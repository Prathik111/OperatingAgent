"""The optional LLM judge, shared by the evaluation harness and the API.

The judge scores *quality* on top of the deterministic checks in
``common.judging``. Fairness is structural, not procedural:

1. **One judge for every track.** A single judge instance (one model, one
   prompt template, temperature 0) is constructed up front and reused for
   every case of every track in a comparison, so no track can be judged by a
   different model or prompt.
2. **Blind judging.** The prompt carries only the goal, the output, and the
   rubric — never the track, the case id, or any run metadata — so the judge
   cannot favour a track it can't see.
3. **Separate reporting.** The judge score never overwrites the deterministic
   pass/fail. It is stored under its own metrics (``judge.overall``,
   ``judge.<criterion>``, ``judge.error``) and rendered in its own tables, so
   the score that depends on a model's opinion is always distinguishable from
   the one that reproduces.
4. **Fail-visible, not fail-silent.** A judge that errors or returns an
   unparseable or partial verdict records ``status="error"`` and a ``None``
   score. Errored judgements are excluded from averages and shown as a count,
   never folded into a pass rate.
5. **Judge overhead is accounted separately.** Tokens and cost consumed by
   judging are recorded on the verdict, not on the judged run, so a track is
   never billed for the judge that scored it.

The judge is optional and off the critical path: without one, evaluation runs
exactly as before on deterministic checks alone.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .judging import OutputCheck

# Default rubric: one criterion per expected artifact. Keyed by the standard
# check names from ``evaluation.suite.ui_artifact_checks``; anything else gets
# the generic criterion. Every criterion is phrased against the goal and the
# output only, which is all the judge is allowed to see.
DEFAULT_RUBRIC: dict[str, str] = {
    "component_code": (
        "The implementation is complete and internally coherent: it defines a "
        "working UI component that matches what the goal asked for, with no "
        "truncated code or placeholder gaps."
    ),
    "component_info": (
        "The documentation describes the generated component accurately: the "
        "inputs it accepts, the states it can be in, and its accessibility "
        "behaviour all correspond to the code actually shown."
    ),
    "pacing": (
        "The timing plan is concrete and usable: it covers the reveal steps in "
        "order, every step has an explicit duration, and the timings form a "
        "sensible progression rather than one arbitrary number."
    ),
    "script": (
        "The narration script matches the pacing plan and the component: it "
        "walks the same steps in the same order and reads as spoken text."
    ),
    "expected_output_contains": (
        "The output contains the information the goal explicitly asked for."
    ),
}
GENERIC_CRITERION = "The output satisfies the goal's requirement for '{name}'."

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator. You will be given a task goal and one "
    "attempted answer. Score each criterion independently. Do not reward "
    "effort, apology, or restating the goal; score only what is present in "
    "the answer. Respond with JSON only, no prose outside the JSON."
)

JUDGE_PROMPT_TEMPLATE = """## Goal
{goal}

## Answer to evaluate
{output}

## Criteria
{criteria}

## Response format
Return a single JSON object:
{{"criteria": [{{"name": "<criterion name>", "pass": true or false, "reason": "<one short sentence>"}}, ...], "overall_comment": "<one short sentence>"}}
Every criterion must appear exactly once. "pass" must be a boolean.
"""


@dataclass(slots=True, frozen=True)
class JudgeCriterion:
    name: str
    description: str


@dataclass(slots=True, frozen=True)
class JudgeCriterionVerdict:
    name: str
    passed: bool
    reason: str = ""


@dataclass(slots=True, frozen=True)
class JudgeVerdict:
    model: str
    status: str  # scored | error
    criteria: tuple[JudgeCriterionVerdict, ...] = ()
    overall_comment: str = ""
    error: str = ""
    tokens: int = 0
    cost: float = 0.0

    @property
    def score(self) -> float | None:
        """Fraction of criteria passed, or None when not cleanly scored."""
        if self.status != "scored" or not self.criteria:
            return None
        return sum(1 for criterion in self.criteria if criterion.passed) / len(self.criteria)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "criteria": [
                {"name": criterion.name, "pass": criterion.passed, "reason": criterion.reason}
                for criterion in self.criteria
            ],
            "overall_comment": self.overall_comment,
            "error": self.error,
            "tokens": self.tokens,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeVerdict:
        return cls(
            model=str(data.get("model", "")),
            status=str(data.get("status", "error")),
            criteria=tuple(
                JudgeCriterionVerdict(
                    name=str(item.get("name", "")),
                    passed=bool(item.get("pass", False)),
                    reason=str(item.get("reason", "")),
                )
                for item in (data.get("criteria") or [])
            ),
            overall_comment=str(data.get("overall_comment", "")),
            error=str(data.get("error", "")),
            tokens=int(data.get("tokens", 0) or 0),
            cost=float(data.get("cost", 0.0) or 0.0),
        )


Complete = Callable[[list[dict[str, str]]], Awaitable[str]]

# Telling a busy provider apart from a broken request, mirroring the native
# loop's heuristic: permanent phrases first, unknown errors stay permanent.
_TEMPORARY_HINTS = (
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "overloaded",
    "capacity",
    "500",
    "502",
    "503",
    "504",
    "internal server error",
    "bad gateway",
    "service unavailable",
)
_PERMANENT_HINTS = (
    "api key",
    "api_key",
    "unauthorized",
    "authentication",
    "401",
    "403",
    "invalid_request",
    "context_length",
    "context length",
    "too large",
)
#: Providers often state their own cool-down, e.g. Groq's
#: "Please try again in 25.3875s" — honour it when present.
_RETRY_WAIT = re.compile(r"try again in ([0-9.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_CAP_SECONDS = 60.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(hint in text for hint in _PERMANENT_HINTS):
        return False
    return any(hint in text for hint in _TEMPORARY_HINTS)


def _retry_delay(exc: BaseException, attempt: int, base_delay: float) -> float:
    """How long to wait before the next judging attempt.

    Prefers the provider's own suggested wait (a token-per-minute window is
    only clearable after it elapses); otherwise exponential backoff.
    """
    match = _RETRY_WAIT.search(str(exc))
    if match:
        return min(float(match.group(1)) + 0.5, _RETRY_DELAY_CAP_SECONDS)
    return min(base_delay * (2**attempt), _RETRY_DELAY_CAP_SECONDS)


@dataclass(slots=True)
class LLMJudge:
    """One model, one prompt template, temperature 0 — reused for every track.

    ``complete`` receives the full message list (system + user) and returns the
    assistant text, so tests inject a scripted responder while production uses
    a real provider adapter.

    Rate-limited calls are retried with backoff rather than recorded as
    ``judge.error``: a 429 says the *provider* is busy, not that the output is
    bad, and scoring it as an error would drag the average down for reasons
    unrelated to quality. A permanent error (bad key, invalid request) fails
    at once.
    """

    complete: Complete
    model: str
    max_output_chars: int = 40_000
    max_retries: int = 5
    retry_base_delay: float = 5.0

    async def judge(self, *, goal: str, output: str, checks: Sequence[OutputCheck]) -> JudgeVerdict:
        criteria = _criteria_from_checks(checks)
        if not criteria:
            return JudgeVerdict(model=self.model, status="error", error="no criteria to judge")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_PROMPT_TEMPLATE.format(
                    goal=goal.strip(),
                    output=(output or "").strip()[: self.max_output_chars],
                    criteria="\n".join(
                        f"- {criterion.name}: {criterion.description}"
                        for criterion in criteria
                    ),
                ),
            },
        ]
        raw: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = await self.complete(messages)
                break
            except Exception as exc:  # noqa: BLE001 - retried or recorded, never raised
                if attempt == self.max_retries or not _is_retryable(exc):
                    tokens, cost = _consume_usage(self.complete)
                    return JudgeVerdict(
                        model=self.model,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        tokens=tokens,
                        cost=cost,
                    )
                delay = _retry_delay(exc, attempt, self.retry_base_delay)
                await asyncio.sleep(delay)
        if raw is None:  # pragma: no cover - the loop always sets one or returns
            raw = ""
        verdict = _parse_verdict(
            raw,
            model=self.model,
            expected_names=tuple(criterion.name for criterion in criteria),
        )
        # Consume on every path so an errored call's usage is never
        # misattributed to the next verdict.
        tokens, cost = _consume_usage(self.complete)
        if tokens or cost:
            verdict = replace(verdict, tokens=tokens, cost=cost)
        return verdict


def _criteria_from_checks(checks: Sequence[OutputCheck]) -> tuple[JudgeCriterion, ...]:
    return tuple(
        JudgeCriterion(
            name=check.name,
            description=DEFAULT_RUBRIC.get(check.name) or GENERIC_CRITERION.format(name=check.name),
        )
        for check in checks
    )


def _parse_verdict(raw: str, *, model: str, expected_names: tuple[str, ...] = ()) -> JudgeVerdict:
    payload = _extract_json(raw)
    if payload is None:
        return JudgeVerdict(
            model=model,
            status="error",
            error="judge response was not parseable JSON",
        )
    if not isinstance(payload, dict):
        return JudgeVerdict(model=model, status="error", error="judge response was not a JSON object")
    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        return JudgeVerdict(model=model, status="error", error="judge response had no criteria list")
    verdicts: list[JudgeCriterionVerdict] = []
    for item in raw_criteria:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        if not isinstance(item.get("pass"), bool):
            return JudgeVerdict(
                model=model,
                status="error",
                error=f"judge criterion {name!r} has a non-boolean pass value",
            )
        verdicts.append(
            JudgeCriterionVerdict(
                name=name,
                passed=item["pass"],
                reason=str(item.get("reason", ""))[:500],
            )
        )
    if not verdicts:
        return JudgeVerdict(model=model, status="error", error="judge response had no named criteria")
    if expected_names:
        # A verdict that does not cover exactly the requested criteria would
        # average over a subset the judge chose itself — refused, not scored.
        returned = [verdict.name for verdict in verdicts]
        missing = [name for name in expected_names if name not in returned]
        unknown = [name for name in returned if name not in expected_names]
        duplicates = sorted({name for name in returned if returned.count(name) > 1})
        if missing or unknown or duplicates:
            problems = []
            if missing:
                problems.append("missing criteria: " + ", ".join(missing))
            if unknown:
                problems.append("unknown criteria: " + ", ".join(unknown))
            if duplicates:
                problems.append("duplicate criteria: " + ", ".join(duplicates))
            return JudgeVerdict(
                model=model,
                status="error",
                error="judge did not score the requested criteria (" + "; ".join(problems) + ")",
            )
    return JudgeVerdict(
        model=model,
        status="scored",
        criteria=tuple(verdicts),
        overall_comment=str(payload.get("overall_comment", ""))[:500],
    )


def _extract_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Tolerate a fenced block or stray prose around the object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _consume_usage(complete: Complete) -> tuple[int, float]:
    tracker = getattr(complete, "usage_tracker", None)
    if tracker is None:
        return 0, 0.0
    return tracker.consume()


class JudgeUsageTracker:
    """Accumulates provider-reported judge usage between calls.

    Public so non-native adapters (e.g. the LangChain-based judge used by the
    API) can attach the same ``usage_tracker`` contract ``LLMJudge`` consumes.
    """

    def __init__(self) -> None:
        self._tokens = 0
        self._cost = 0.0

    def add(self, tokens: int, cost: float) -> None:
        self._tokens += tokens
        self._cost += cost

    def consume(self) -> tuple[int, float]:
        tokens, cost = self._tokens, self._cost
        self._tokens, self._cost = 0, 0.0
        return tokens, cost


async def _registry_complete(
    registry: Any,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    usage_tracker: JudgeUsageTracker | None = None,
) -> str:
    """Adapt an ``agent_native`` ModelRegistry into the judge's callable.

    Streams with no tools and concatenates the text events, mirroring how the
    native loop consumes a provider, and forwards provider-reported usage to
    the tracker so judging overhead is billed to the judge, not the judged run.
    The ``agent_native`` import is deliberately lazy: this module lives in
    ``common``, which must stay importable without the agent packages.
    """
    from agent_native.models.base import StreamType

    model = registry.get(model_name)
    provider = registry.get_provider(model)
    chunks: list[str] = []
    async for event in provider.stream(
        messages,
        [],
        model,
        temperature=temperature,
    ):
        if event.type is StreamType.TEXT and event.data.get("text"):
            chunks.append(str(event.data["text"]))
        elif event.type is StreamType.USAGE and usage_tracker is not None:
            usage = event.data
            tokens = int(usage.get("input_tokens", 0) or 0) + int(
                usage.get("output_tokens", 0) or 0
            )
            cost = model.cost_of(
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
            )
            usage_tracker.add(tokens, cost)
    return "".join(chunks)


def judge_from_registry(registry: Any, model_name: str) -> LLMJudge:
    """Build the single judge instance a comparison run must share."""
    if model_name not in registry.list_model_names():
        raise KeyError(
            f"judge model {model_name!r} is not registered "
            f"(available: {', '.join(registry.list_model_names())})"
        )
    usage_tracker = JudgeUsageTracker()

    async def complete(messages: list[dict[str, str]]) -> str:
        return await _registry_complete(
            registry, model_name, messages, usage_tracker=usage_tracker
        )

    complete.usage_tracker = usage_tracker  # type: ignore[attr-defined]
    return LLMJudge(complete=complete, model=model_name)


@dataclass(slots=True)
class JudgeSummary:
    """Aggregate over one track's verdicts — kept separate from pass rates."""

    judged: int = 0
    errors: int = 0
    scores: list[float] = field(default_factory=list)
    by_criterion: dict[str, list[bool]] = field(default_factory=dict)
    tokens: int = 0
    cost: float = 0.0

    @property
    def average(self) -> float | None:
        return (sum(self.scores) / len(self.scores)) if self.scores else None

    @classmethod
    def from_verdicts(cls, verdicts: Sequence[JudgeVerdict | None]) -> JudgeSummary:
        summary = cls()
        for verdict in verdicts:
            if verdict is None:
                continue
            summary.tokens += verdict.tokens
            summary.cost += verdict.cost
            if verdict.status != "scored":
                summary.errors += 1
                continue
            summary.judged += 1
            score = verdict.score
            if score is not None:
                summary.scores.append(score)
            for criterion in verdict.criteria:
                summary.by_criterion.setdefault(criterion.name, []).append(criterion.passed)
        return summary

    def criterion_pass_rate(self, name: str) -> float | None:
        outcomes = self.by_criterion.get(name)
        return (sum(1 for passed in outcomes if passed) / len(outcomes)) if outcomes else None


def summarize_judgments(
    verdicts_by_case: Mapping[str, JudgeVerdict | None],
) -> dict[str, Any]:
    """Render a track's judge summary as a JSON-friendly dict."""
    summary = JudgeSummary.from_verdicts(list(verdicts_by_case.values()))
    criteria = {
        name: (sum(1 for passed in outcomes if passed) / len(outcomes)) if outcomes else None
        for name, outcomes in sorted(summary.by_criterion.items())
    }
    return {
        "judged": summary.judged,
        "errors": summary.errors,
        "average_score": summary.average,
        "criteria_pass_rate": criteria,
        "tokens": summary.tokens,
        "cost": summary.cost,
    }

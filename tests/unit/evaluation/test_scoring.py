from __future__ import annotations

import json

import pytest
from agent_native.models.base import Model, ModelRegistry, StreamEvent, StreamType
from evaluation.scoring import (
    JudgeVerdict,
    LLMJudge,
    judge_from_registry,
    summarize_judgments,
)
from evaluation.suite import ui_artifact_checks

GOOD_VERDICT = json.dumps(
    {
        "criteria": [
            {"name": "component_code", "pass": True, "reason": "complete"},
            {"name": "component_info", "pass": True, "reason": "accurate"},
            {"name": "pacing", "pass": True, "reason": "concrete"},
            {"name": "script", "pass": False, "reason": "mismatched steps"},
        ],
        "overall_comment": "mostly solid",
    }
)


def _judge(responses: list[str | Exception]) -> tuple[LLMJudge, list[list[dict[str, str]]]]:
    seen: list[list[dict[str, str]]] = []

    async def complete(messages: list[dict[str, str]]) -> str:
        seen.append(messages)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return LLMJudge(complete=complete, model="test-judge"), seen


async def test_judge_parses_clean_json_and_scores_criteria() -> None:
    judge, _ = _judge([GOOD_VERDICT])
    verdict = await judge.judge(goal="build a card", output="...output...", checks=ui_artifact_checks())
    assert verdict.status == "scored"
    assert verdict.score == pytest.approx(0.75)  # 3 of 4 artifact criteria passed
    assert verdict.criteria[3].name == "script"


async def test_judge_refuses_partial_criterion_coverage() -> None:
    partial = json.dumps(
        {"criteria": [{"name": "component_code", "pass": True}]}
    )
    judge, _ = _judge([partial])
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert verdict.status == "error"
    assert "missing criteria" in verdict.error
    assert "component_info" in verdict.error


async def test_judge_retries_rate_limit_and_recovers() -> None:
    calls: list[int] = []

    async def complete(messages: list[dict[str, str]]) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(
                "429 rate limit reached; Please try again in 0.01s"
            )
        return GOOD_VERDICT

    judge = LLMJudge(complete=complete, model="m", retry_base_delay=0.001)
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert len(calls) == 2
    assert verdict.status == "scored"  # a busy provider is not a quality verdict


async def test_judge_gives_up_after_max_retries() -> None:
    calls: list[int] = []

    async def complete(messages: list[dict[str, str]]) -> str:
        calls.append(1)
        raise RuntimeError("Error code: 429 - rate limit exceeded")

    judge = LLMJudge(
        complete=complete, model="m", max_retries=3, retry_base_delay=0.001
    )
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert len(calls) == 4  # initial attempt + 3 retries
    assert verdict.status == "error"
    assert "429" in verdict.error


async def test_judge_does_not_retry_permanent_errors() -> None:
    calls: list[int] = []

    async def complete(messages: list[dict[str, str]]) -> str:
        calls.append(1)
        raise RuntimeError("Error code: 401 - invalid api key")

    judge = LLMJudge(complete=complete, model="m", retry_base_delay=0.001)
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert len(calls) == 1
    assert verdict.status == "error"
    assert "401" in verdict.error


async def test_judge_parses_fenced_json_and_stray_prose() -> None:
    judge, _ = _judge([f"sure!\n```json\n{GOOD_VERDICT}\n```\nthanks"])
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert verdict.status == "scored"


async def test_judge_fails_visible_on_garbage() -> None:
    judge, _ = _judge(["not json at all"])
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert verdict.status == "error"
    assert verdict.score is None
    assert "parseable" in verdict.error


async def test_judge_fails_visible_on_provider_error() -> None:
    judge, _ = _judge([RuntimeError("provider down")])
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert verdict.status == "error"
    assert "provider down" in verdict.error


async def test_judge_prompt_is_blind_to_track_and_case() -> None:
    judge, seen = _judge([GOOD_VERDICT])
    await judge.judge(goal="build a card", output="the answer", checks=ui_artifact_checks())
    prompt = "\n".join(message["content"] for message in seen[0])
    assert "build a card" in prompt
    assert "the answer" in prompt
    for leaked in ("native", "langgraph", "case_id", "track"):
        assert leaked not in prompt.lower()


async def test_judge_without_checks_records_error() -> None:
    judge, _ = _judge([GOOD_VERDICT])
    verdict = await judge.judge(goal="g", output="o", checks=())
    assert verdict.status == "error"
    assert "no criteria" in verdict.error


def test_verdict_roundtrip_preserves_score() -> None:
    from evaluation.scoring import JudgeCriterionVerdict

    verdict = JudgeVerdict(
        model="m",
        status="scored",
        criteria=(
            JudgeCriterionVerdict(name="component_code", passed=True, reason="ok"),
            JudgeCriterionVerdict(name="script", passed=False, reason="off-step"),
        ),
        overall_comment="fine",
        tokens=12,
        cost=0.5,
    )
    restored = JudgeVerdict.from_dict(verdict.to_dict())
    assert restored.score == verdict.score == pytest.approx(0.5)
    assert restored.criteria == verdict.criteria
    assert restored.tokens == 12 and restored.cost == 0.5


async def test_summary_separates_scored_from_errored() -> None:
    scored = JudgeVerdict.from_dict(
        {
            "model": "m",
            "status": "scored",
            "criteria": [
                {"name": "component_code", "pass": True},
                {"name": "script", "pass": False},
            ],
            "tokens": 120,
            "cost": 0.002,
        }
    )
    errored = JudgeVerdict(model="m", status="error", error="boom", tokens=10, cost=0.001)
    summary = summarize_judgments({"a": scored, "b": errored, "c": None})
    assert summary["judged"] == 1
    assert summary["errors"] == 1
    assert summary["average_score"] == pytest.approx(0.5)
    assert summary["criteria_pass_rate"]["component_code"] == pytest.approx(1.0)
    assert summary["criteria_pass_rate"]["script"] == pytest.approx(0.0)
    assert summary["tokens"] == 130
    assert summary["cost"] == pytest.approx(0.003)


class _ScriptedProvider:
    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)

    async def stream(self, messages, tools, model, temperature=0.0, **kwargs):
        text = self._texts.pop(0)
        for chunk in (text[:5], text[5:]):
            if chunk:
                yield StreamEvent(StreamType.TEXT, {"text": chunk})
        yield StreamEvent(
            StreamType.USAGE, {"input_tokens": 100, "output_tokens": 40, "cached_tokens": 0}
        )


def _registry(*texts: str) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register_provider("groq", _ScriptedProvider(*texts))
    registry.register_model(
        "judge-model",
        Model(
            provider="groq",
            model_id="judge-model",
            input_price_per_million=1.0,
            output_price_per_million=2.0,
        ),
    )
    return registry


async def test_judge_from_registry_streams_and_accounts_usage() -> None:
    judge = judge_from_registry(_registry(GOOD_VERDICT), "judge-model")
    assert judge.model == "judge-model"
    verdict = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert verdict.status == "scored"
    assert verdict.tokens == 140
    assert verdict.cost == pytest.approx((100 * 1.0 + 40 * 2.0) / 1_000_000)


async def test_errored_call_usage_is_not_misattributed_to_next_verdict() -> None:
    judge = judge_from_registry(_registry("garbage not json", GOOD_VERDICT), "judge-model")
    first = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    second = await judge.judge(goal="g", output="o", checks=ui_artifact_checks())
    assert first.status == "error"
    assert first.tokens == 140  # the failed call still carries its own usage
    assert second.status == "scored"
    assert second.tokens == 140  # not 280 — no leakage across verdicts


def test_judge_from_registry_rejects_unknown_model() -> None:
    with pytest.raises(KeyError, match="not registered"):
        judge_from_registry(_registry("x"), "missing-model")

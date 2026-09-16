from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_langgraph.orchestrator.langgraph_agent import UsageTracker
from common.config import (
    AgentConfig,
    BehaviourConfig,
    CheckpointConfig,
    ExecutionConfig,
    LLMConfig,
    PromptConfig,
    SandboxConfig,
    ToolPermissionConfig,
    TracingConfig,
)


def _config(prices: tuple[float, float] = (0.0, 0.0)) -> AgentConfig:
    return AgentConfig(
        llm=LLMConfig(
            provider="groq",
            model="m",
            api_key="k",
            input_price_per_million=prices[0],
            output_price_per_million=prices[1],
        ),
        execution=ExecutionConfig(),
        sandbox=SandboxConfig(),
        permissions=ToolPermissionConfig(),
        checkpoint=CheckpointConfig(),
        tracing=TracingConfig(),
        behaviour=BehaviourConfig(),
        prompts=PromptConfig(
            planner_prompt=Path("prompts/planner.txt"),
            verifier_prompt=Path("prompts/verifier.txt"),
            responder_prompt=Path("prompts/responder.txt"),
        ),
    )


def _response(usages: list[dict]):
    generations = [
        [SimpleNamespace(message=SimpleNamespace(usage_metadata=usage))]
        for usage in usages
    ]
    return SimpleNamespace(generations=generations)


def test_tracker_accumulates_usage_across_calls() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(_response([{"input_tokens": 100, "output_tokens": 40}]))
    tracker.on_llm_end(_response([{"input_tokens": 60, "output_tokens": 20}]))
    assert tracker.calls == 2
    assert tracker.prompt_tokens == 160
    assert tracker.completion_tokens == 60
    assert tracker.total_tokens == 220


def test_tracker_tolerates_malformed_responses() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(SimpleNamespace(generations=None))
    tracker.on_llm_end(SimpleNamespace(generations=[[object()]]))
    tracker.on_llm_end(_response([{}]))
    assert tracker.calls == 0
    assert tracker.total_tokens == 0


def test_tracker_prices_usage_only_when_configured() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(_response([{"input_tokens": 1_000_000, "output_tokens": 500_000}]))
    assert tracker.cost(_config()) is None  # no pricing = unknown, not $0
    priced = tracker.cost(_config((0.6, 2.4)))
    assert priced == pytest.approx(0.6 + 2.4 * 0.5)  # 1M in + 0.5M out


def test_tracker_reads_llm_output_token_usage_fallback() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(
        SimpleNamespace(
            generations=[],
            llm_output={"token_usage": {"prompt_tokens": 11, "completion_tokens": 4}},
        )
    )
    assert tracker.calls == 1
    assert tracker.total_tokens == 15


def test_tracker_reads_prompt_completion_aliases_from_response_metadata() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(
        SimpleNamespace(
            generations=[[SimpleNamespace(message=SimpleNamespace(
                usage_metadata={},
                response_metadata={"token_usage": {"prompt_tokens": 8, "completion_tokens": 3}},
            ))]],
            llm_output={},
        )
    )
    assert tracker.calls == 1
    assert tracker.prompt_tokens == 8
    assert tracker.completion_tokens == 3


def test_tracker_keeps_provider_total_when_only_total_tokens_is_reported() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(
        SimpleNamespace(
            generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata={}))]],
            usage_metadata={"total_tokens": 17},
            llm_output={},
        )
    )
    assert tracker.calls == 1
    assert tracker.total_tokens == 17

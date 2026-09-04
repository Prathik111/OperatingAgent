from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from api.config import ApiSettings
from common.agent import AgentRunResult
from common.enums import AgentTrack, RunStatus
from evaluation.execution import EvaluationEnvironment, run_suite_for_tracks
from evaluation.suite import EvaluationCase, EvaluationSuite


@dataclass
class FakeOrchestrator:
    output: str
    calls: list = field(default_factory=list)

    async def run(self, task):
        self.calls.append(task)
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output=self.output,
            duration_ms=12.3,
            llm_calls=1,
            tool_calls=0,
            total_tokens=42,
            cost=0.5,
            metadata={"source": self.output},
        )


@pytest.mark.asyncio
async def test_run_suite_uses_both_tracks_and_track_specific_sessions() -> None:
    native = FakeOrchestrator("native done")
    langgraph = FakeOrchestrator("langgraph done")
    environment = EvaluationEnvironment(
        settings=ApiSettings(),
        orchestrators={
            AgentTrack.NATIVE: native,
            AgentTrack.LANGGRAPH: langgraph,
        },
        native_runtime=None,
        native_pool=None,
    )
    suite = EvaluationSuite(
        id="demo",
        cases=(EvaluationCase(id="case-1", goal="say hello"),),
    )

    results = await run_suite_for_tracks(suite, environment)

    assert [result.track for result in results] == ["native", "langgraph"]
    assert [result.case_id for result in results] == ["case-1", "case-1"]
    assert native.calls[0].thread_id.startswith("eval-native-")
    assert langgraph.calls[0].thread_id.startswith("eval-langgraph-")
    assert native.calls[0].metadata["working_directory"] == "."

"""End-to-end: the main business flow, hermetically.

One task in, one ``AgentRunResult`` out, through the whole compiled graph —
planner -> executor -> verifier -> responder — via the public
``LangGraphAgent.run`` entry point. The LLM and MCP gateway are stubbed so this
tier stays fast and deterministic and can run on every commit; the same flow
against real Groq + Langfuse lives in ``tests/e2e/live/test_agent_flow_live.py``.

Few tests by design (the top of the pyramid): the happy path, the two failure
modes a caller must be able to rely on, and event emission. Anything more
granular belongs in the unit or integration tier.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_langgraph.graph.state import AgentPlan, PlanStep
from agent_langgraph.orchestrator.langgraph_agent import LangGraphAgent
from common.agent import AgentTask
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent
from common.tools import ToolCallResult

from tests.support.langgraph import (
    StubModel,
    StubModelProvider,
    StubPromptManager,
    StubToolRegistry,
    build_agent_config,
    make_plan,
    make_step,
)

#: The answer the stubbed responder produces; asserted by reference rather than
#: as a literal so changing the stub default cannot silently break this tier.
FINAL_ANSWER = "Echoed 'hello' and summarised the result."


def build_agent(
    *,
    config=None,
    model: StubModel | None = None,
    tool_registry: StubToolRegistry | None = None,
) -> tuple[LangGraphAgent, StubToolRegistry]:
    config = config or build_agent_config(require_human_approval=False)
    registry = tool_registry or StubToolRegistry(
        default=ToolCallResult(success=True, output="echo:hello", error=None)
    )
    agent = LangGraphAgent(
        config,
        tool_registry=registry,
        model_provider=StubModelProvider(model or StubModel(answer=FINAL_ANSWER)),
        prompt_manager=StubPromptManager(),
    )
    return agent, registry


def make_task(**overrides: Any) -> AgentTask:
    fields: dict[str, Any] = {
        "id": "task-1",
        "goal": "Echo hello",
        "thread_id": "thread-42",
        "track": AgentTrack.LANGGRAPH,
        "metadata": {"user_id": "user-7", "feature": "smoke-test"},
    }
    fields.update(overrides)
    return AgentTask(**fields)


# ---------------------------------------------------------------------------
# Happy path — plan, call a tool, verify, answer
# ---------------------------------------------------------------------------


async def test_agent_completes_a_task_end_to_end() -> None:
    agent, registry = build_agent()
    result = await agent.run(make_task())

    assert result.status is RunStatus.COMPLETED
    assert result.output == FINAL_ANSWER
    # DEFAULT_PLAN has one tool-backed step and one reasoning step.
    assert result.tool_calls == 1
    assert registry.calls == [("echo_tool", {"text": "hello"})]
    assert result.duration_ms >= 0
    assert result.metadata.get("checkpoint_id")


async def test_agent_can_run_without_streaming_graph_states() -> None:
    agent, registry = build_agent(
        config=build_agent_config(
            require_human_approval=False,
            stream=False,
        )
    )

    result = await agent.run(make_task())

    assert result.status is RunStatus.COMPLETED
    assert result.output == FINAL_ANSWER
    assert registry.calls == [("echo_tool", {"text": "hello"})]


async def test_agent_runs_a_two_phase_task_end_to_end() -> None:
    """A find-then-act goal runs two phases. The planner returns an
    investigate-only plan (``requires_remediation=True``); once it is exhausted
    the graph goes back through the planner for a remediation phase before
    responding. The tool therefore runs once per phase, and the run still
    terminates on its own — the monotonic phase advance is what bounds it.

    ``registry.calls`` is the ground truth for "the tool ran twice";
    ``result.tool_calls`` reflects only the final (remediation) plan, so it is
    not the multi-phase signal here.
    """
    investigate_then_fix = AgentPlan(
        summary="inspect, then fix",
        reasoning="a find-then-act goal needs a follow-up plan",
        steps=[
            PlanStep(id=1, description="inspect the workspace", tool_name="echo_tool",
                     arguments={"text": "hello"}),
        ],
        requires_remediation=True,
    )
    agent, registry = build_agent(
        model=StubModel(plan=investigate_then_fix, answer=FINAL_ANSWER)
    )

    result = await agent.run(make_task(goal="check the workspace for bugs and fix them"))

    assert result.status is RunStatus.COMPLETED
    assert result.output == FINAL_ANSWER
    # One tool call in the investigate phase, one in the remediate phase.
    assert registry.calls == [
        ("echo_tool", {"text": "hello"}),
        ("echo_tool", {"text": "hello"}),
    ]


async def test_agent_emits_state_and_finished_events() -> None:
    agent, _ = build_agent()
    events: list[AgentEvent] = []

    await agent.run(make_task(), on_event=events.append)

    types = [e.type for e in events]
    assert "state" in types
    assert types[-1] == "finished"
    assert events[-1].payload["status"] == RunStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# Failure modes a caller relies on: an honest result, never an exception
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agent_reports_tool_failure_as_failed_result() -> None:
    """A tool that keeps failing exhausts the retry budget and the responder
    reports FAILED — no exception escapes ``run``."""
    registry = StubToolRegistry(
        default=ToolCallResult(success=False, output=None, error="permanent error")
    )
    agent, _ = build_agent(
        config=build_agent_config(require_human_approval=False, retry_attempts=0),
        tool_registry=registry,
    )

    result = await agent.run(make_task())
    assert result.status is RunStatus.FAILED
    assert result.metadata.get("error")


@pytest.mark.regression
async def test_agent_converts_node_exception_into_failed_result() -> None:
    """If a node raises (here the planner's model), ``run`` catches it, emits an
    error event, and returns FAILED rather than propagating to the caller."""
    agent, _ = build_agent(model=StubModel(structured_error=RuntimeError("planner down")))
    events: list[AgentEvent] = []

    result = await agent.run(make_task(), on_event=events.append)
    assert result.status is RunStatus.FAILED
    assert "error" in [e.type for e in events]


@pytest.mark.regression
async def test_agent_tolerates_a_failing_event_listener() -> None:
    """A bad listener must not fail the run — observability is not load-bearing."""
    agent, _ = build_agent()

    def bad_listener(_event: AgentEvent) -> None:
        raise RuntimeError("listener boom")

    result = await agent.run(make_task(), on_event=bad_listener)
    assert result.status is RunStatus.COMPLETED


async def test_agent_continues_same_thread_with_prior_transcript() -> None:
    agent, _ = build_agent()
    model = agent._model_provider.model

    first = await agent.run(make_task(id="turn-1", goal="first turn"))
    second_task = make_task(
        id="turn-2",
        goal="second turn",
        execution_mode="continue",
    )
    second = await agent.run(second_task)

    assert first.status is RunStatus.COMPLETED
    assert second.status is RunStatus.COMPLETED
    responder_messages = [getattr(item, "content", "") for item in model.invocations[-1]]
    assert any("second turn" in content for content in responder_messages)


async def test_agent_resumes_an_interrupted_checkpoint() -> None:
    config = build_agent_config(require_human_approval=True, enable_interrupts=True)
    model = StubModel(
        plan=make_plan(make_step(tool_name="delete_file", arguments={"path": "x"}))
    )
    agent, _ = build_agent(config=config, model=model)
    task = make_task(id="interrupt-1", goal="delete the file")

    interrupted = await agent.run(task)
    assert interrupted.status is RunStatus.INTERRUPTED

    task.execution_mode = "resume"
    task.resume_value = {"approved": True}
    resumed = await agent.run(task)

    assert resumed.status is RunStatus.COMPLETED
    assert resumed.output == FINAL_ANSWER

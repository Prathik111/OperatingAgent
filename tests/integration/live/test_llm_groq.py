"""Live integration: the real Groq provider behind ``ModelProvider``.

Proves the contract the unit tests fake. ``tests/unit/agent_langgraph/test_planner.py``
asserts the planner asks for ``AgentPlan`` with ``method="json_schema"`` and gets
an ``AgentPlan`` back from a stub — that is only meaningful if a real model
actually honours it, which is what these tests check.

Assertions are **structural, not textual**: a live model's wording is not
reproducible, so we assert on shape (a plan with steps, a boolean verdict, tool
names drawn from the offered set) and never on exact strings.
"""

from __future__ import annotations

from agent_langgraph.graph.state import AgentPlan
from agent_langgraph.nodes.planner import PlannerNode
from agent_langgraph.nodes.verifier import _Verdict
from agent_langgraph.runtime.model_provider import ModelProvider
from common.enums import TaskStatus

from tests.support.langgraph import (
    StubToolRegistry,
    build_context,
    build_runtime,
    make_state,
    make_tool_info,
)
from tests.support.live import InlinePromptManager, build_live_config, groq_model

# ---------------------------------------------------------------------------
# The provider itself
# ---------------------------------------------------------------------------


def test_model_provider_builds_a_real_groq_model() -> None:
    model = ModelProvider(build_live_config()).get_model()
    assert type(model).__name__ == "ChatGroq"
    assert model.model_name == groq_model()


async def test_real_model_answers_a_plain_prompt() -> None:
    """The cheapest possible proof that the credentials and model id work."""
    model = ModelProvider(build_live_config()).get_model()
    message = await model.ainvoke("Reply with the single word: pong")
    assert isinstance(message.content, str)
    assert message.content.strip()


# ---------------------------------------------------------------------------
# Structured output — the capability the planner and verifier depend on
# ---------------------------------------------------------------------------


async def test_real_model_produces_a_valid_agent_plan() -> None:
    """``method="json_schema"`` must yield a schema-valid ``AgentPlan``. If the
    configured model lacks structured-output support this is where it shows."""
    model = ModelProvider(build_live_config()).get_model()
    structured = model.with_structured_output(AgentPlan, method="json_schema")

    plan = await structured.ainvoke(
        [
            ("system", InlinePromptManager().planner()),
            ("human", "Goal: echo the phrase 'hello' and then count its words."),
        ]
    )

    assert isinstance(plan, AgentPlan)
    assert plan.steps, "a live plan must contain at least one step"
    assert all(step.description for step in plan.steps)


async def test_real_model_produces_a_structured_verdict() -> None:
    model = ModelProvider(build_live_config()).get_model()
    structured = model.with_structured_output(_Verdict)

    verdict = await structured.ainvoke(
        [
            ("system", InlinePromptManager().verifier()),
            ("human", "Step intent: echo the word 'hello'. Output produced: 'hello'. Did it succeed?"),
        ]
    )

    assert isinstance(verdict.success, bool)
    assert verdict.success is True, "an obviously-correct step should verify"


# ---------------------------------------------------------------------------
# Nodes driven by the real model
# ---------------------------------------------------------------------------


async def test_planner_node_against_real_model_offers_real_tools() -> None:
    """The planner hints the available tools into its prompt; a live model should
    pick a tool name from that set rather than inventing one."""
    config = build_live_config()
    registry = StubToolRegistry(
        tools=[make_tool_info("echo", "Return the given text unchanged."),
               make_tool_info("word_count", "Count whitespace-separated words in text.")]
    )
    context = build_context(
        config,
        model_provider=ModelProvider(config),
        tool_registry=registry,
        prompt_manager=InlinePromptManager(),
    )

    delta = await PlannerNode(
        make_state(goal="Echo the phrase 'hello world', then count its words."),
        build_runtime(context),
    )

    plan = delta["plan"]
    assert isinstance(plan, AgentPlan)
    assert plan.steps
    assert delta["status"] is TaskStatus.PLANNING
    # Any tool the model chose must be one we actually offered.
    chosen = {s.tool_name for s in plan.steps if s.tool_name}
    assert chosen <= {"echo", "word_count"}, f"model invented tools: {chosen}"

"""Integration: how the graph is wired and configured, short of running a task.

Covers the seam between ``GraphFactory`` (topology) and ``LangGraphAgent``
(compilation + the invocation config that carries Langfuse trace attributes).
Everything here is hermetic — stubs stand in for the LLM and MCP, and no
credentials are present, so tracing runs its no-op path.

The *behaviour* of the individual routing functions lives in
``tests/unit/agent_langgraph/test_routing.py``; the whole-task flow lives in
``tests/e2e/test_agent_flow.py``. This module is only about wiring.
"""

from __future__ import annotations

from typing import Any

from agent_langgraph.graph.builder import GraphFactory
from agent_langgraph.graph.constants import (
    ERROR_HANDLER,
    EXECUTOR,
    PHASE_TRANSITION,
    PLANNER,
    RESPONDER,
    VERIFIER,
)
from agent_langgraph.orchestrator.langgraph_agent import LangGraphAgent
from common.agent import AgentTask
from common.config import MetadataConfig
from common.enums import AgentTrack, TaskStatus
from common.tools import ToolCallResult
from langgraph.checkpoint.memory import MemorySaver

from tests.support.langgraph import (
    StubModel,
    StubModelProvider,
    StubPromptManager,
    StubToolRegistry,
    build_agent_config,
    build_context,
)

ALL_NODES = {PLANNER, EXECUTOR, VERIFIER, RESPONDER, ERROR_HANDLER, PHASE_TRANSITION}


def build_agent(*, config=None, model: StubModel | None = None) -> LangGraphAgent:
    config = config or build_agent_config(require_human_approval=False)
    return LangGraphAgent(
        config,
        tool_registry=StubToolRegistry(
            default=ToolCallResult(success=True, output="echo:hello", error=None)
        ),
        model_provider=StubModelProvider(model or StubModel()),
        prompt_manager=StubPromptManager(),
    )


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
# Topology
# ---------------------------------------------------------------------------


def test_unconditional_edges_are_wired() -> None:
    """START->planner, executor->verifier and responder->END are fixed edges,
    and every node is registered. The conditional routes are asserted
    behaviourally in the routing unit tests."""
    compiled = GraphFactory().create_graph().compile()
    drawable = compiled.get_graph()

    assert ALL_NODES <= set(drawable.nodes)

    edges = {(e.source, e.target) for e in drawable.edges}
    assert any(target == PLANNER for _src, target in edges)  # START -> planner
    assert (EXECUTOR, VERIFIER) in edges
    assert any(src == RESPONDER for src, _target in edges)  # responder -> END
    # Planner results either execute or respond directly. Only an exhausted,
    # verified plan reaches the phase transition for a replan or response.
    assert (PLANNER, RESPONDER) in edges
    assert (PLANNER, PHASE_TRANSITION) not in edges
    assert (VERIFIER, PHASE_TRANSITION) in edges
    assert (PHASE_TRANSITION, PLANNER) in edges
    assert (PHASE_TRANSITION, RESPONDER) in edges


async def test_compiled_graph_runs_to_completion() -> None:
    """Drive the compiled graph directly (not via the orchestrator) with a real
    Runtime context of stubs — proves the topology plus ``context_schema``
    wiring executes planner -> executor -> verifier -> responder."""
    config = build_agent_config(require_human_approval=False)
    compiled = GraphFactory().create_graph().compile(checkpointer=MemorySaver())

    final_state = await compiled.ainvoke(
        {"goal": "Echo hello", "messages": [], "current_step": 0, "retry_count": 0},
        config={"configurable": {"thread_id": "graph-test"}},
        context=build_context(config),
    )

    assert final_state["status"] is TaskStatus.COMPLETED
    assert final_state["messages"][-1].content
    # Both plan steps are marked verified/completed on the happy path.
    assert all(step.verified for step in final_state["plan"].steps)


# ---------------------------------------------------------------------------
# Invocation config (the Langfuse trace attributes)
# ---------------------------------------------------------------------------


def test_invocation_config_builds_trace_attributes() -> None:
    agent_config = build_agent_config(
        checkpoint_namespace="tenant-a",
        metadata=MetadataConfig(
            tags={"environment": "test"},
            custom={"deployment": "local"},
        ),
    )
    config = build_agent(config=agent_config)._invocation_config(make_task())

    assert config["configurable"]["thread_id"] == "thread-42"
    assert config["configurable"]["checkpoint_ns"] == "tenant-a"
    assert config["run_name"] == "agent-run:langgraph"
    metadata = config["metadata"]
    assert metadata["langfuse_session_id"] == "thread-42"
    assert metadata["task_id"] == "task-1"
    assert metadata["langfuse_user_id"] == "user-7"
    assert "track:langgraph" in metadata["langfuse_tags"]
    assert "feature:smoke-test" in metadata["langfuse_tags"]
    assert "environment:test" in metadata["langfuse_tags"]
    assert metadata["deployment"] == "local"


def test_invocation_config_callbacks_empty_when_tracing_disabled() -> None:
    """No credentials in this tier, so no Langfuse callback may be attached."""
    config = build_agent()._invocation_config(make_task())
    assert config["callbacks"] == []


def test_invocation_config_omits_user_id_when_absent() -> None:
    config = build_agent()._invocation_config(make_task(metadata={}))
    assert "langfuse_user_id" not in config["metadata"]
    assert config["metadata"]["langfuse_tags"] == ["track:langgraph"]


def test_recursion_limit_scales_with_iteration_budget() -> None:
    agent = build_agent(
        config=build_agent_config(require_human_approval=False, max_iterations=5)
    )
    assert agent._invocation_config(make_task())["recursion_limit"] == 5 * 4


# ---------------------------------------------------------------------------
# Compile caching
# ---------------------------------------------------------------------------


async def test_graph_is_compiled_once_and_reused() -> None:
    agent = build_agent()
    first = await agent._compile()
    assert first is await agent._compile()
    await agent.aclose()
    assert agent._compiled is None

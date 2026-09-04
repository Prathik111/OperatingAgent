"""Live end-to-end: the whole agent against real Groq, Langfuse and MCP.

The apex of the pyramid, and deliberately a single test. One
``LangGraphAgent.run`` drives planner -> executor -> verifier -> responder with
a real LLM at every step, real MCP tools over the real protocol, and a real
Langfuse trace — then asserts everything worth asserting about that one run.
Splitting it into several tests would mean several billed multi-call LLM runs
for no extra coverage.

Everything it touches is genuine:

* **LLM** — real Groq (~4 calls: plan, verify each step, respond).
* **Tools** — a real in-process FastMCP server; ``TOOL_CALLS`` records actual
  invocations, so a passing assertion means the model really called the tool.
* **Tracing** — real Langfuse; the run returns the trace id it created.

Assertions are behavioural, never textual: a live model's phrasing is not
reproducible, so we check the tools ran with the right inputs and the answer
mentions the derived fact, not that it used particular words.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_langgraph.mcp_adapter import MCPAdapter
from agent_langgraph.orchestrator.langgraph_agent import LangGraphAgent
from agent_langgraph.runtime.model_provider import ModelProvider
from agent_langgraph.runtime.tool_registry import ToolRegistry
from common.agent import AgentTask
from common.enums import AgentTrack, RunStatus
from common.events import AgentEvent

from tests.support.live import InlinePromptManager, build_live_config

PHRASE = "hello from groq"
THREAD_ID = "pytest-live-thread"


@pytest.fixture
def tool_server():
    """A real FastMCP server plus the log of calls that actually reached it."""
    fastmcp = pytest.importorskip("fastmcp")

    server = fastmcp.FastMCP(name="pytest-live-tools")
    calls: list[tuple[str, dict[str, Any]]] = []

    @server.tool
    def echo(text: str) -> str:
        """Return the given text unchanged."""
        calls.append(("echo", {"text": text}))
        return text

    @server.tool
    def word_count(text: str) -> int:
        """Count the whitespace-separated words in the given text."""
        calls.append(("word_count", {"text": text}))
        return len(text.split())

    return server, calls


async def test_agent_completes_a_real_task_end_to_end(tool_server) -> None:
    server, tool_calls = tool_server
    config = build_live_config()

    agent = LangGraphAgent(
        config,
        tool_registry=ToolRegistry(MCPAdapter(server)),
        model_provider=ModelProvider(config),
        prompt_manager=InlinePromptManager(),
    )
    assert agent._tracer.enabled, "live e2e must run with tracing on"

    task = AgentTask(
        id="pytest-live-1",
        goal=f"Echo the phrase '{PHRASE}', then report how many words it contains.",
        thread_id=THREAD_ID,
        track=AgentTrack.LANGGRAPH,
        metadata={"user_id": "pytest-live-user", "feature": "live-e2e"},
    )

    events: list[AgentEvent] = []
    result = await agent.run(task, on_event=events.append)

    # --- the run succeeded -------------------------------------------------
    assert result.status is RunStatus.COMPLETED, (
        f"live run failed: {result.metadata.get('error')!r}"
    )
    assert result.output and result.output.strip()

    # --- the tools genuinely executed -------------------------------------
    invoked = {name for name, _args in tool_calls}
    assert "echo" in invoked, f"the model never called echo; called {invoked}"
    assert any(PHRASE in str(args.get("text", "")) for _n, args in tool_calls), (
        f"no tool received the phrase under test: {tool_calls}"
    )
    assert result.tool_calls >= 1

    # --- the answer reflects the work, not just the goal -------------------
    # "hello from groq" is three words; a real end-to-end run should surface it.
    assert "3" in result.output or "three" in result.output.lower(), (
        f"final answer does not report the word count: {result.output!r}"
    )

    # --- the run was observable -------------------------------------------
    types = [e.type for e in events]
    assert "state" in types
    assert types[-1] == "finished"

    trace_id = result.metadata.get("langfuse_trace_id")
    assert trace_id, "a live traced run must report the Langfuse trace id"
    assert isinstance(trace_id, str) and len(trace_id) >= 16

"""LangGraph implementation of the agent orchestration interface."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from agent_langgraph.checkpoint_factory import CheckpointFactory
from agent_langgraph.graph.builder import GraphFactory
from agent_langgraph.mcp_adapter import MCPAdapter
from agent_langgraph.runtime.context import (
    AgentContext,
    ModelProviderLike,
    PromptManagerLike,
    ToolRegistryLike,
)
from agent_langgraph.runtime.model_provider import ModelProvider
from agent_langgraph.runtime.prompt_manager import PromptManager
from agent_langgraph.runtime.tool_registry import ToolRegistry
from agent_langgraph.tracing.tracer import Tracer
from common.agent import AgentRunResult, AgentTask
from common.approvals import ApprovalHandler
from common.config import AgentConfig
from common.enums import RunStatus, TaskStatus
from common.events import AgentEvent
from common.interfaces import IAgentOrchestrator
from common.risk import RiskClassifier
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, messages_to_dict
from langgraph.types import Command

log = logging.getLogger(__name__)

_TERMINAL_STATUS = {
    TaskStatus.COMPLETED: RunStatus.COMPLETED,
    TaskStatus.FAILED: RunStatus.FAILED,
    TaskStatus.INTERRUPTED: RunStatus.INTERRUPTED,
}


class UsageTracker(BaseCallbackHandler):
    """Accumulates provider-reported token usage across a graph run.

    LangChain fires ``on_llm_end`` for every model call with the generation's
    ``usage_metadata`` (input/output tokens) — the same numbers the Langfuse
    handler would capture. Recording them here lets the run's receipt carry
    real token totals even when Langfuse is not configured, which is what the
    evaluation dashboard reads. Purely observational: it never raises.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            def record(usage: Any) -> None:
                prompt = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                completion = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                if not prompt and not completion:
                    # A few providers report only the aggregate. Keep
                    # the total visible rather than dropping the call.
                    completion = usage.get("total_tokens", 0) or 0
                self.prompt_tokens += int(prompt)
                self.completion_tokens += int(completion)
                self.calls += 1

            captured = False
            for generations in response.generations or []:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None) or {}
                    if not usage and message is not None:
                        response_metadata = getattr(message, "response_metadata", {}) or {}
                        usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
                    if usage:
                        record(usage)
                        captured = True
            if captured:
                return
            # Response-level shapes (usage_metadata or the llm_output aggregate)
            # are read once, outside the per-generation loop, and only count
            # when no generation carried its own numbers, so provider choice
            # does not silently turn usage into an empty dash or double-count.
            usage = getattr(response, "usage_metadata", None) or {}
            if usage:
                record(usage)
                return
            raw = getattr(response, "llm_output", None) or {}
            usage = raw.get("token_usage") if isinstance(raw, dict) else None
            usage = usage or (raw.get("usage") if isinstance(raw, dict) else None)
            if isinstance(usage, dict):
                record(usage)
        except Exception as exc:  # noqa: BLE001 - usage capture must never break a run
            log.debug("could not read usage from an LLM result: %s", exc)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, config: AgentConfig) -> float | None:
        """Price the captured usage, or None when pricing is not configured."""
        llm = config.llm
        if llm.input_price_per_million <= 0 and llm.output_price_per_million <= 0:
            return None
        return (
            self.prompt_tokens * llm.input_price_per_million
            + self.completion_tokens * llm.output_price_per_million
        ) / 1_000_000


class LangGraphAgent(IAgentOrchestrator):
    """
    LangGraphAgent orchestrates the agent's reasoning and execution flow using LangGraph.
    It constructs a StateGraph based on the agent's plan and executes it using the GraphExecutor.

    Tracing: the Langfuse ``CallbackHandler`` is attached to each invocation, so
    every node, LLM generation (with model + token usage) and LangChain tool call
    is captured automatically with the correct observation types. Direct MCP tool
    calls bypass LangChain and are traced by the ExecutorNode's own ``tool`` span.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        tool_registry: ToolRegistryLike | None = None,
        model_provider: ModelProviderLike | None = None,
        prompt_manager: PromptManagerLike | None = None,
        risk_classifier: RiskClassifier | None = None,
        tracer: Tracer | None = None,
        approval_handler: ApprovalHandler | None = None,
        mcp_gateway_command: str | None = None,
        mcp_gateway_args: list[str] | None = None,
    ):
        self.config = config
        self.checkpoint_factory = CheckpointFactory(config)
        self.graph_factory = GraphFactory()

        # Dependencies are injectable so tests can supply fakes; by default they
        # are built from config/env.
        self._tracer = tracer or Tracer(config.tracing)
        self._model_provider = model_provider or ModelProvider(config)
        self._risk_classifier = risk_classifier or RiskClassifier()
        self._prompt_manager = prompt_manager or PromptManager(config.prompts)
        self._approval_handler = approval_handler
        self._tool_registry: ToolRegistryLike
        if tool_registry is None:
            import sys

            self._tool_registry = ToolRegistry(
                MCPAdapter.from_stdio(
                    mcp_gateway_command or sys.executable,
                    mcp_gateway_args or ["-m", "gateway_server"],
                ),
                permissions=config.permissions,
                sandbox=config.sandbox,
            )
        else:
            self._tool_registry = tool_registry

        self._compiled: Any | None = None
        self._checkpointer_context: AbstractAsyncContextManager[Any] | None = None
        self._compile_lock = asyncio.Lock()
        self._auto_approve_all = False

    @property
    def auto_approve_all(self) -> bool:
        """Whether approval gates are skipped (BLOCKED is still rejected)."""
        return self._auto_approve_all

    def set_auto_approve_all(self, enabled: bool) -> bool:
        """Turn the allow-all opt-in on or off live. Returns the new state."""
        self._auto_approve_all = bool(enabled)
        return self._auto_approve_all

    # -- graph lifecycle ---------------------------------------------------

    async def _compile(self) -> Any:
        """Open the configured saver, then compile once and reuse the graph."""
        async with self._compile_lock:
            if self._compiled is not None:
                return self._compiled

            checkpointer_context = self.checkpoint_factory.open_checkpointer()
            checkpointer = await checkpointer_context.__aenter__()
            try:
                compiled = self.graph_factory.create_graph().compile(
                    checkpointer=checkpointer
                )
            except BaseException as exc:
                await checkpointer_context.__aexit__(
                    type(exc), exc, exc.__traceback__
                )
                raise

            self._checkpointer_context = checkpointer_context
            self._compiled = compiled
            return compiled

    async def aclose(self) -> None:
        """Close the checkpointer and MCP subprocess resources."""
        async with self._compile_lock:
            checkpointer_context = self._checkpointer_context
            self._checkpointer_context = None
            self._compiled = None
            if checkpointer_context is not None:
                await checkpointer_context.__aexit__(None, None, None)
        close_registry = getattr(self._tool_registry, "aclose", None)
        if close_registry is not None:
            await close_registry()

    async def reconfigure(self, config: AgentConfig) -> None:
        """Apply model/runtime settings to future invocations.

        The compiled graph and checkpoint store are independent of the model
        provider, so active checkpoints remain valid. Existing invocations keep
        their context; subsequent tasks receive the new provider and config.
        """
        async with self._compile_lock:
            self.config = config
            self._model_provider = ModelProvider(config)
            self._prompt_manager = PromptManager(config.prompts)

    def _build_context(
        self,
        task: AgentTask,
        config: AgentConfig,
        model_provider: ModelProviderLike,
        prompt_manager: PromptManagerLike,
        on_event: Callable[[AgentEvent], Awaitable[None] | None] | None = None,
    ) -> AgentContext:
        return AgentContext(
            model_provider=model_provider,
            tool_registry=self._tool_registry,
            risk_classifier=self._risk_classifier,
            prompt_manager=prompt_manager,
            tracer=self._tracer,
            config=config,
            approval_handler=self._approval_handler,
            auto_approve_all=self._auto_approve_all,
            task_id=task.id,
            event_sink=on_event,
            completed_tool_calls=task.completed_tool_calls,
            workspace=str(
                task.metadata.get("workspace")
                or task.metadata.get("working_directory")
                or ""
            )
            or None,
        )

    def _invocation_config(
        self, task: AgentTask, config: AgentConfig | None = None
    ) -> dict[str, Any]:
        """Build the LangChain config, including all Langfuse trace attributes.

        - ``run_name`` becomes the trace name, so traces are filterable by
          something descriptive rather than an opaque id.
        - ``langfuse_session_id`` is the thread id, which groups every turn of a
          conversation into one session in the Sessions view.
        - ``langfuse_user_id`` enables per-user filtering and cost attribution
          (taken from task metadata when the caller supplies it).
        - ``langfuse_tags`` allow per-track / per-feature dashboards.
        """
        config = config or self.config
        handler = self._tracer.callback_handler()

        tags = [f"track:{task.track.value}"]
        tags.extend(f"{key}:{value}" for key, value in config.metadata.tags.items())
        feature = task.metadata.get("feature")
        if feature:
            tags.append(f"feature:{feature}")

        metadata: dict[str, Any] = {
            str(key): _langfuse_metadata_value(value)
            for key, value in config.metadata.custom.items()
            if value is not None
        }
        metadata.update({
            "langfuse_session_id": task.thread_id,
            "langfuse_tags": list(dict.fromkeys(tags)),
            "langfuse_trace_name": f"agent-run:{task.track.value}",
            "task_id": task.id,
        })
        workspace = task.metadata.get("workspace") or task.metadata.get("working_directory")
        if workspace:
            metadata["workspace"] = str(workspace)
        user_id = task.metadata.get("user_id")
        if user_id:
            metadata["langfuse_user_id"] = str(user_id)
        environment = task.metadata.get("langfuse_environment")
        if environment:
            metadata["langfuse_environment"] = str(environment)
        version = task.metadata.get("langfuse_version")
        if version:
            metadata["langfuse_version"] = str(version)

        invocation: dict[str, Any] = {
            "configurable": {
                "thread_id": task.thread_id,
                "checkpoint_ns": (
                    task.resume_checkpoint_namespace
                    or config.checkpoint.namespace
                ),
            },
            "callbacks": [handler] if handler is not None else [],
            "run_name": f"agent-run:{task.track.value}",
            "metadata": metadata,
            "recursion_limit": config.execution.max_iterations * 4,
        }
        if task.resume_checkpoint_id:
            invocation["configurable"]["checkpoint_id"] = task.resume_checkpoint_id
        metadata["execution_mode"] = task.execution_mode
        return invocation

    # -- orchestration -----------------------------------------------------

    async def run(
        self,
        task: AgentTask,
        on_event: Callable[[AgentEvent], Awaitable[None] | None] | None = None,
    ) -> AgentRunResult:
        """Execute the graph for ``task`` and return the run result.

        Traces are flushed before returning so short-lived processes (CLI runs,
        serverless invocations) don't drop buffered spans.
        """
        graph = await self._compile()
        # Capture all configuration-dependent dependencies together. A later
        # reconfigure() may replace these fields, but this invocation must use
        # one coherent provider/prompt/config snapshot from start to finish.
        async with self._compile_lock:
            config = self.config
            model_provider = self._model_provider
            prompt_manager = self._prompt_manager
            invocation = self._invocation_config(task, config)
        # Direct observers are best-effort, but the API service's callback is
        # authoritative because it persists the execution history. Keep the
        # latter's fail-closed behavior while isolating ordinary listeners.
        event_sink = on_event
        if on_event is not None and not getattr(on_event, "_authoritative", False):
            async def event_sink(event: AgentEvent) -> None:
                await self._emit(on_event, event)
        handler = next(iter(invocation["callbacks"]), None)
        usage_tracker = UsageTracker()
        invocation["callbacks"].append(usage_tracker)

        async def attach_usage(result: AgentRunResult) -> AgentRunResult:
            """Put captured usage on the receipt and persist it as an llm_call.

            The service persists llm_call events into the llm_calls table (the
            Postgres metrics view and the in-memory backend both sum it), and
            the evaluation path falls back to the receipt metadata — so both
            surfaces see the same totals.
            """
            if usage_tracker.calls:
                await self._emit(
                    event_sink,
                    AgentEvent(
                        type="llm_call",
                        payload={
                            "node_name": "langgraph_run",
                            "provider": config.llm.provider,
                            "model": config.llm.model,
                            "prompt_tokens": usage_tracker.prompt_tokens,
                            "completion_tokens": usage_tracker.completion_tokens,
                            "cost": usage_tracker.cost(config),
                        },
                    ),
                )
                result.llm_calls = usage_tracker.calls
                result.total_tokens = usage_tracker.total_tokens
                cost = usage_tracker.cost(config)
                if cost is not None:
                    result.cost = cost
                result.metadata["llm_calls"] = usage_tracker.calls
                result.metadata["total_tokens"] = usage_tracker.total_tokens
                if cost is not None:
                    result.metadata["cost"] = cost
            return result

        handler = next(iter(invocation["callbacks"][:-1]), None)

        started = time.perf_counter()
        final_state: dict[str, Any] | None = None
        checkpoint_id: str | None = None

        try:
            if task.execution_mode == "resume":
                graph_input: Any = (
                    Command(resume=task.resume_value)
                    if task.resume_value is not None
                    else None
                )
            else:
                graph_input = {
                    "goal": task.goal,
                    "messages": [HumanMessage(content=task.goal)],
                    "current_step": 0,
                    "retry_count": 0,
                    # The (pure) retry router reads the replan budget from
                    # state, so seed it here from the run's configuration.
                    "max_replans": config.execution.max_replans,
                }
                if task.execution_mode == "continue":
                    # A new conversational turn keeps the transcript but must
                    # not inherit the previous turn's active plan/verdict.
                    graph_input.update(
                        {
                            "plan": None,
                            "workflow_phase": None,
                            "verification_success": None,
                            "verification_reason": None,
                            "last_error": None,
                            "status": None,
                        }
                    )
            context = self._build_context(
                task, config, model_provider, prompt_manager, event_sink
            )
            if config.execution.stream:
                async for state in graph.astream(
                    graph_input,
                    config=invocation,
                    context=context,
                    stream_mode="values",
                ):
                    final_state = state
                    await self._emit_state(on_event, state)
            else:
                invoked_state = await graph.ainvoke(
                    graph_input,
                    config=invocation,
                    context=context,
                )
                if invoked_state is None:
                    raise RuntimeError("graph completed without a final state")
                final_state = invoked_state
                await self._emit_state(on_event, invoked_state)
        except Exception as exc:
            log.exception("graph execution failed for task %s", task.id)
            await self._emit(on_event, AgentEvent(type="error", payload={"error": str(exc)}))
            checkpoint_id = await self._latest_checkpoint_id(graph, invocation)
            return await attach_usage(self._result(
                status=RunStatus.FAILED,
                output=None,
                started=started,
                state=final_state,
                error=str(exc),
                handler=handler,
                checkpoint_id=checkpoint_id,
            ))
        finally:
            # Buffered spans must be delivered even when the run failed.
            self._tracer.flush()

        checkpoint_id = await self._latest_checkpoint_id(graph, invocation)

        result = await attach_usage(self._result(
            status=_run_status(final_state),
            output=_final_output(final_state),
            started=started,
            state=final_state,
            error=(final_state or {}).get("last_error"),
            handler=handler,
            checkpoint_id=checkpoint_id,
        ))
        await self._emit(
            on_event,
            AgentEvent(
                type="finished",
                payload={
                    "status": result.status.value,
                    "final_message": result.output,
                    "trace_id": result.metadata.get("langfuse_trace_id"),
                },
            ),
        )
        return result

    # -- helpers -----------------------------------------------------------

    def _result(
        self,
        *,
        status: RunStatus,
        output: str | None,
        started: float,
        state: dict[str, Any] | None,
        error: str | None,
        handler: Any | None,
        checkpoint_id: str | None,
    ) -> AgentRunResult:
        """Assemble the run result, linking it back to its Langfuse trace.

        ``langfuse_trace_id`` is what lets a stored run be joined to its trace
        (the AGENT_RUNS.langfuse_trace_id column in the database design).
        Token and cost totals start at zero and are filled in by
        ``attach_usage`` from the run's captured usage, so the receipt carries
        real numbers even without Langfuse.
        """
        plan = (state or {}).get("plan")
        steps = getattr(plan, "steps", []) or []
        tool_calls = sum(1 for s in steps if s.tool_name)

        metadata: dict[str, Any] = {}
        if error:
            metadata["error"] = error
        trace_id = getattr(handler, "last_trace_id", None) if handler is not None else None
        if trace_id:
            metadata["langfuse_trace_id"] = trace_id
        if checkpoint_id:
            metadata["checkpoint_id"] = checkpoint_id

        return AgentRunResult(
            status=status,
            output=output,
            duration_ms=(time.perf_counter() - started) * 1000,
            llm_calls=0,
            tool_calls=tool_calls,
            total_tokens=0,
            cost=0.0,
            metadata=metadata,
        )

    @staticmethod
    async def _latest_checkpoint_id(graph: Any, invocation: dict[str, Any]) -> str | None:
        """Read the latest persisted checkpoint for the current invocation."""
        try:
            # ``checkpoint_ns`` is LangGraph's subgraph selector, not an
            # application tenant label.  The top-level graph is addressed by
            # thread id (and an optional checkpoint id) alone.
            configurable = dict(invocation.get("configurable", {}))
            configurable.pop("checkpoint_ns", None)
            lookup_config = {"configurable": configurable}
            snapshot = await graph.aget_state(lookup_config)
            configurable = getattr(snapshot, "config", {}).get("configurable", {})
            checkpoint_id = configurable.get("checkpoint_id")
            return str(checkpoint_id) if checkpoint_id else None
        except Exception as exc:  # noqa: BLE001 - metadata lookup is best effort
            # Checkpoint reads are observability metadata; they must not turn a
            # completed graph run into a failed one.
            log.warning("could not read latest checkpoint metadata: %s", exc)
            return None

    @staticmethod
    async def _emit_state(
        on_event: Callable[[AgentEvent], Awaitable[None] | None] | None,
        state: dict[str, Any],
    ) -> None:
        await LangGraphAgent._emit(
            on_event,
            AgentEvent(
                type="state",
                payload={
                    "status": _status_value(state.get("status")),
                    "current_step": state.get("current_step"),
                    "goal": state.get("goal"),
                    "messages": _message_history(state.get("messages", [])),
                },
            ),
        )

    @staticmethod
    async def _emit(
        on_event: Callable[[AgentEvent], Awaitable[None] | None] | None,
        event: AgentEvent,
    ) -> None:
        """Deliver an event, tolerating sync or async callbacks."""
        if on_event is None:
            return
        try:
            outcome = on_event(event)
            if outcome is not None and hasattr(outcome, "__await__"):
                await outcome
        except Exception as exc:  # noqa: BLE001 - callback isolation boundary
            log.warning("event callback raised: %s", exc)


def _status_value(status: Any) -> str | None:
    return status.value if isinstance(status, TaskStatus) else status


def _run_status(state: dict[str, Any] | None) -> RunStatus:
    status = (state or {}).get("status")
    if isinstance(status, TaskStatus):
        return _TERMINAL_STATUS.get(status, RunStatus.INTERRUPTED)
    return RunStatus.INTERRUPTED


def _final_output(state: dict[str, Any] | None) -> str | None:
    """The responder's answer is the last message on the final state."""
    messages = (state or {}).get("messages") or []
    if not messages:
        return None
    content = getattr(messages[-1], "content", None)
    return content if isinstance(content, str) else None


def _message_history(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert checkpointed LangChain messages to durable JSON event data."""
    try:
        return messages_to_dict(messages)
    except (TypeError, ValueError):
        return [
            {"type": type(message).__name__, "content": str(getattr(message, "content", message))}
            for message in messages
        ]


def _langfuse_metadata_value(value: Any) -> str:
    """Encode custom metadata for Langfuse v4's string-valued metadata field."""
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
    return rendered[:200]

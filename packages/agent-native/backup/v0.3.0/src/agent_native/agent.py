"""NativeAgent - the hand-written Plan-and-Execute + ReAct orchestrator.

Composes Planner, ReactExecutor, Reflector, Verifier, ContextCompactor,
ApprovalGateway, SandboxManager, MCP clients, RiskClassifier, TaskRepository
and TracingService. Exposes its own run(task, on_event) - this package's own
entry protocol (decision #10: local protocol + CLI; packages/api stays
untouched and adapts later if ever needed).

Loop:
  plan -> for each step: execute (ReAct w/ risk gate + approval + verification)
  on DENIED/BLOCKED/VERIFY_FAIL/MAX_CALLS_EXCEEDED -> Reflector.replan
    (bounded by max_replans, decision #1); budget exhausted -> FAILED with
    an explicit REPLAN_BUDGET_EXHAUSTED event.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from .approval import ApprovalGateway, emit_event
from .compactor import ContextCompactor
from .config import SandboxConfig, Settings
from .events import (
    AGENT_FINISHED,
    PLANNING_FAILED,
    PLANNING_STARTED,
    PLANNING_SUCCEEDED,
    REPLAN_BUDGET_EXHAUSTED,
    RUN_FAILED,
    AgentEvent,
)
from .llm import LLMClient
from .mcp import MCPClient, Spawner, StdioMCPClient
from .planner import Planner, PlanningError, tool_catalog_text
from .executor import ReactExecutor, StepOutcome
from .reflector import Reflector, ReplanBudgetExhausted
from .repository import TaskRepository
from .risk import RiskClassifier
from .sandbox import SandboxManager
from .tracing import TracingService
from .types import AgentRunResult, AgentTask, Plan, RunStatus, StepOutcomeStatus
from .verifier import Verifier

EventCallback = Callable[[AgentEvent], Awaitable[None] | None]


class NativeAgent:
    def __init__(
        self,
        planner: Planner,
        executor: ReactExecutor,
        reflector: Reflector,
        repository: TaskRepository,
        llm: LLMClient,
        mcp: MCPClient,
        risk: RiskClassifier,
        verifier: Verifier,
        compactor: ContextCompactor,
        approval: ApprovalGateway | None,
        sandbox: SandboxManager | None = None,
        tracing: TracingService | None = None,
        *,
        settings: Settings | None = None,
        tools: list | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.reflector = reflector
        self.repository = repository
        self.llm = llm
        self.mcp = mcp
        self.risk = risk
        self.verifier = verifier
        self.compactor = compactor
        self.approval = approval
        self.sandbox = sandbox
        self.tracing = tracing or TracingService()
        self.settings = settings or Settings()
        self.max_replans = self.settings.max_replans
        self._manually_provided_tools = tools

    async def run(
        self,
        task: AgentTask,
        on_event: EventCallback | None = None,
    ) -> AgentRunResult:
        start = time.monotonic()
        self.risk.start_session(task.id)
        llm_calls = 0
        tool_calls = 0
        tokens = 0
        replans = 0
        output: str | None = None
        failure: str | None = None

        try:
            await self.repository.save_task(task)
            tools = self._manually_provided_tools or await self.mcp.list_tools()

            await emit_event(on_event, AgentEvent(
                kind=PLANNING_STARTED, task_id=task.id,
                payload={"goal": task.goal},
            ))
            with self.tracing.span("llm", {"model": self.settings.model_name()}):
                try:
                    plan = await self.planner.plan(task, tools)
                except PlanningError as e:
                    await emit_event(on_event, AgentEvent(
                        kind=PLANNING_FAILED, task_id=task.id,
                        payload={"reason": str(e)},
                    ))
                    raise
            await emit_event(on_event, AgentEvent(
                kind=PLANNING_SUCCEEDED, task_id=task.id,
                payload={"steps": len(plan.steps)},
            ))

            plan_context = tool_catalog_text(tools)
            step_index = 0
            while step_index < len(plan.steps):
                step = plan.steps[step_index]
                with self.tracing.span("llm", {"model": self.settings.model_name()}):
                    outcome = await self.executor.execute_step(task, step, tools, plan_context)
                llm_calls += outcome.llm_calls
                tool_calls += outcome.tool_calls
                tokens += outcome.tokens

                if outcome.status == StepOutcomeStatus.SUCCESS:
                    output = outcome.output or output
                    step_index += 1
                    continue

                reason = outcome.reason or outcome.status.value
                if replans >= self.max_replans:
                    failure = f"replan budget exhausted ({self.max_replans}) after: {reason}"
                    await emit_event(on_event, AgentEvent(
                        kind=REPLAN_BUDGET_EXHAUSTED, task_id=task.id,
                        payload={"reason": reason, "max_replans": self.max_replans},
                    ))
                    break
                replans += 1
                try:
                    with self.tracing.span("llm", {"model": self.settings.model_name()}):
                        plan = await self.reflector.replan(task, plan, reason, tools)
                except ReplanBudgetExhausted as e:
                    failure = str(e)
                    await emit_event(on_event, AgentEvent(
                        kind=REPLAN_BUDGET_EXHAUSTED, task_id=task.id,
                        payload={"reason": str(e)},
                    ))
                    break
                plan_context = tool_catalog_text(tools)
                step_index = 0  # new plan supersedes; restart (bounded by max_replans)

            duration_ms = (time.monotonic() - start) * 1000.0
            if failure is not None:
                result = AgentRunResult(
                    status=RunStatus.FAILED, output=output, duration_ms=duration_ms,
                    llm_calls=llm_calls, tool_calls=tool_calls, total_tokens=tokens,
                    replans=replans, failure_reason=failure,
                    metadata={"task_id": task.id},
                )
                await emit_event(on_event, AgentEvent(
                    kind=RUN_FAILED, task_id=task.id,
                    payload={"status": result.status.value, "failure_reason": failure},
                ))
            else:
                result = AgentRunResult(
                    status=RunStatus.COMPLETED, output=output, duration_ms=duration_ms,
                    llm_calls=llm_calls, tool_calls=tool_calls, total_tokens=tokens,
                    replans=replans, metadata={"task_id": task.id},
                )
            await self.repository.save_run_result(result)
            await emit_event(on_event, AgentEvent(
                kind=AGENT_FINISHED, task_id=task.id,
                payload={"status": result.status.value, "output": output},
            ))
            return result
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000.0
            result = AgentRunResult(
                status=RunStatus.FAILED, output=None, duration_ms=duration_ms,
                llm_calls=llm_calls, tool_calls=tool_calls, total_tokens=tokens,
                replans=replans, failure_reason=str(e),
                metadata={"task_id": task.id},
            )
            await self.repository.save_run_result(result)
            await emit_event(on_event, AgentEvent(
                kind=RUN_FAILED, task_id=task.id, payload={"failure_reason": str(e)},
            ))
            return result
        finally:
            self.risk.end_session(task.id)


def build_agent(
    settings: Settings,
    llm: LLMClient | None = None,
    mcp: MCPClient | None = None,
    repository: TaskRepository | None = None,
    sandbox: SandboxManager | None = None,
    approval: ApprovalGateway | None = None,
    on_event: EventCallback | None = None,
    tools: list | None = None,
) -> NativeAgent:
    """Wiring factory: defaults the dependencies consumers don't override."""
    from .config import is_host_allowed
    from .llm import build_llm

    llm = llm or build_llm(settings)
    deny_egress = settings.sandbox.egress.lower() == "deny"
    risk = RiskClassifier(
        allowlist_net_hosts=(lambda host: is_host_allowed(host, settings.sandbox))
        if deny_egress else None,
    )
    sandbox = sandbox or (SandboxManager(settings.sandbox) if settings.sandbox.enabled else None)
    verifier = Verifier()
    compactor = ContextCompactor(settings.token_budget)
    approval = approval or ApprovalGateway(timeout_s=settings.approval_timeout_s, on_event=on_event)
    repository = repository or _memory_repository()

    client = mcp or StdioMCPClient(
        name="file",
        spawner=direct_spawner(settings, "file")
        if "file" in settings.mcp_server_commands else None,
    )
    executor = ReactExecutor(
        llm=llm, mcp=client, risk=risk, verifier=verifier,
        compactor=compactor, approval=approval,
        max_calls_per_step=settings.max_calls_per_step, on_event=on_event,
    )
    planner = Planner(llm=llm, repository=repository)
    reflector = Reflector(llm=llm, repository=repository,
                          max_replans=settings.max_replans, on_event=on_event)
    agent = NativeAgent(
        planner=planner, executor=executor, reflector=reflector,
        repository=repository, llm=llm, mcp=client, risk=risk,
        verifier=verifier, compactor=compactor, approval=approval,
        sandbox=sandbox, settings=settings, tools=tools,
    )
    return agent


def _resolve_command(cmd: list[str]) -> list[str]:
    """Resolve a bare executable name the way a shell would, checking the
    venv Scripts dir (the interpreter's own directory) as a fallback PATH."""
    import shutil
    import sys

    exe = cmd[0]
    if "/" in exe or "\\" in exe or Path(exe).suffix:
        return cmd
    found = shutil.which(exe)
    if found:
        return [found, *cmd[1:]]
    scripts = Path(sys.executable).parent
    for candidate in (exe, exe + ".exe", exe + ".cmd", exe + ".bat"):
        if (scripts / candidate).is_file():
            return [str(scripts / candidate), *cmd[1:]]
    return cmd


def sandboxed_spawner(
    settings: Settings,
    sandbox: SandboxManager,
    task_id: str,
    server_name: str,
) -> Spawner:
    """Async spawner that runs an MCP server inside the task's sandbox
    container (decision #6: same container, same deny-all egress policy)."""

    async def _spawn() -> tuple[object, object, object]:
        import asyncio

        session = sandbox.get_session(task_id) or sandbox.create_session(task_id)
        cmd = _resolve_command(settings.mcp_server_commands.get(server_name, ["python", "-m", server_name]))
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", session.container_id, *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        return proc.stdout, proc.stdin, proc

    return _spawn


def direct_spawner(settings: Settings, server_name: str) -> Spawner:
    """Async spawner running the server as a direct child process."""

    async def _spawn() -> tuple[object, object, object]:
        import asyncio

        cmd = _resolve_command(list(settings.mcp_server_commands.get(server_name, ["python", "-m", server_name])))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        return proc.stdout, proc.stdin, proc

    return _spawn


def new_task(goal: str, thread_id: str = "") -> AgentTask:
    return AgentTask(id=uuid.uuid4().hex[:16], goal=goal, thread_id=thread_id)


def _memory_repository() -> TaskRepository:
    from .repository import InMemoryTaskRepository

    return InMemoryTaskRepository()
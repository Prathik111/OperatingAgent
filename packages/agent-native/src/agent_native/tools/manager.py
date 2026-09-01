"""The tool dispatcher: the one gate every tool call goes through.

Given a tool call the model made, it does the same four things every time, in
order: find the tool, check the arguments, check permission, then run it. If any
of the first three fails, it never runs the tool - it hands back a `ToolResult`
that says why. The loop treats that failure exactly like any other result: it
goes back into the conversation for the model to read. That is the whole reason
a denied or malformed call can't crash a run.

Those four steps are split across two methods - `authorize` (the first three) and
`run_authorized` (the last) - with `execute` doing both, which is what almost
every caller wants. The split exists because a turn's calls can run at the same
time but their permission prompts cannot: three tools all asking at once is a mess
on a terminal. So the loop authorizes in order, then runs what was approved
together. Nothing skips the gate; it just opens in two moves.

Every call also gets a timeout. Concurrency without one means a single hung tool
holds up the whole turn, and a turn that never ends is worse than a tool that
failed.

One more thing happens in `run_authorized`: a tool marked `SANDBOX` is handed to
the container runner instead of being called here. That is a single branch, in one
place, because every tool call already comes through this gate - which is the
payoff of having one. If the container isn't available the branch falls through and
the tool runs normally, so a machine without Docker still works; `sandbox.py` says
which mode a run ended up in.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import ArgumentChecker, ExecutionMode, Tool, ToolRegistry, ToolResult

#: How long any one tool may take when it doesn't say otherwise. Generous, because
#: this is a backstop against hanging - not a performance budget.
DEFAULT_TIMEOUT_SECONDS = 120.0


class ToolManager:
    """Finds, checks, gates and runs tools. Nothing runs a tool but this."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: Any,
        permissions: Any,
        argument_checker: ArgumentChecker | None = None,
        max_output_chars: int = 12_000,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sandbox: Any = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._permissions = permissions
        self._args = argument_checker or ArgumentChecker()
        self._max_output = max_output_chars
        self._timeout = timeout_seconds
        #: Where SANDBOX-marked tools run. None means "nowhere else" - they run
        #: here like everything else, which is the old behaviour.
        self.sandbox = sandbox

    async def execute(self, tool_call: Any, context: Any) -> ToolResult:
        """Run one tool call, or explain why it didn't run."""
        refusal = await self.authorize(tool_call, context)
        if refusal is not None:
            return refusal
        return await self.run_authorized(tool_call, context)

    async def authorize(self, tool_call: Any, context: Any) -> ToolResult | None:
        """Everything that happens before a tool runs.

        Returns None when the call may go ahead, or the result explaining why it
        may not. May wait on the user, so callers should do this in a predictable
        order rather than all at once.
        """
        tool = self._find_tool(tool_call.name)
        if tool is None:
            return ToolResult(False, error=f"Unknown tool: {tool_call.name!r}")

        ok, reason = self._args.validate(tool.definition.input_schema, tool_call.arguments)
        if not ok:
            return ToolResult(False, error=f"Invalid arguments: {reason}")

        allowed, why = await self._check_permission(context, tool, tool_call)
        if not allowed:
            return ToolResult(False, error=f"Not allowed: {why}")
        return None

    async def run_authorized(self, tool_call: Any, context: Any) -> ToolResult:
        """Run a call `authorize` already approved. Never hangs, never raises."""
        tool = self._find_tool(tool_call.name)
        if tool is None:  # vanished between authorizing and running; can't happen, so say so plainly
            return ToolResult(False, error=f"Unknown tool: {tool_call.name!r}")

        seconds = self.timeout_for(tool)
        try:
            result = await asyncio.wait_for(
                self._invoke(tool, tool_call, context, seconds), timeout=seconds
            )
        except TimeoutError:
            # A timeout is a result like any other: the model reads it and can try
            # something narrower instead of the run dying here.
            return ToolResult(False, error=f"Timed out after {seconds:g}s")
        except Exception as exc:  # a crashing tool is a result, not a run-ender
            return ToolResult(False, error=f"Tool crashed: {type(exc).__name__}: {exc}")

        return self._limit_output(result)

    async def _invoke(self, tool: Tool, tool_call: Any, context: Any, seconds: float) -> ToolResult:
        """Actually run the tool - in the container if it's marked for one.

        The sandbox answering None means it couldn't (no Docker, no working folder,
        no command to run), never that the tool failed. So the fallback is to run
        it here, exactly as an agent with no sandbox at all would.
        """
        if self.sandbox is not None and self._is_sandboxed(tool):
            result = await self.sandbox.run(tool, tool_call.arguments, context, seconds)
            if result is not None:
                return result
        return await tool.execute(tool_call.arguments, context)

    def _is_sandboxed(self, tool: Tool) -> bool:
        return tool.definition.permissions.execution_mode is ExecutionMode.SANDBOX

    def timeout_for(self, tool: Tool) -> float:
        """This tool's own limit if it set one, else the shared default."""
        own = getattr(tool.definition, "timeout_seconds", 0.0) or 0.0
        return float(own) if own > 0 else self._timeout

    # -- the pieces, named so the flow reads top to bottom ------------------
    def _find_tool(self, name: str) -> Tool | None:
        return self._registry.find(name)

    async def _check_permission(self, context: Any, tool: Tool, tool_call: Any) -> tuple:
        """Ask the policy; if it says 'ask', ask the user (reusing any saved grant)."""
        from ..permissions import PermissionDecision, PermissionRequest

        decision = self._policy.check(context, tool.definition, tool_call.arguments)

        if decision.result == PermissionDecision.DENY:
            return False, decision.reason or "denied by policy"

        if decision.result == PermissionDecision.ASK:
            request = PermissionRequest(
                call_id=tool_call.id,
                tool=tool.definition.full_name,
                arguments=dict(tool_call.arguments),
                preview=self._preview(tool, tool_call.arguments),
                reason=decision.reason,
            )
            allowed = await self._permissions.ask(request, context.session.id, context.run_id)
            return allowed, ("allowed by user" if allowed else "denied by user")

        return True, decision.reason or "allowed by policy"

    def _preview(self, tool: Tool, arguments: dict) -> str:
        """The line the user approves - what will happen, and where.

        Where matters as much as what. "Run rm -rf build" is a different decision
        depending on whether it lands in a throwaway container or on the machine,
        and the person clicking yes is the one who should be told which.
        """
        preview = tool.preview(arguments)
        if self.sandbox is not None and self._is_sandboxed(tool):
            return f"{preview} [in the sandbox container]"
        if tool.definition.permissions.execution_mode is ExecutionMode.HOST_PROCESS:
            return f"{preview} [on your machine]"
        return preview

    def _limit_output(self, result: ToolResult) -> ToolResult:
        """Keep a huge tool output from blowing up the next model call."""
        if result.output and len(result.output) > self._max_output:
            kept = result.output[: self._max_output]
            result.output = kept + f"\n... [truncated {len(result.output) - self._max_output} characters]"
            result.truncated = True
        return result

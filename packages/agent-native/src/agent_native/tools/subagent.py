"""Handing a smaller job to a helper.

The point of this isn't speed. It's keeping the main conversation clean: a search
that reads thirty files should cost the main conversation one paragraph, not
thirty files. The helper does the reading in a conversation of its own and hands
back a summary, and only that summary lands in the parent's transcript.

Four things had to be got right, and each one is a trap if it isn't:

**The helper's messages must not leak into the parent's conversation.** They share
a session - that's how the helper's work shows up on the parent's event stream -
but the parent reloads its conversation from the database by session id, so if the
helper saved its thirty file reads there, the next turn would read them all back
and the feature would have achieved precisely nothing. So the helper's messages
are filed under a session of their own (`_HelperMessages` below), which keeps them
out of the parent's conversation while still keeping them.

**The helper cannot delegate or fan out.** It gets a tool list with both
delegation tools removed, so a helper can't hire a helper that hires a helper, nor
open a fan-out wave of its own. Depth-limiting would work too; removing the tools
is simpler and needs no bookkeeping.

**Stopping the parent stops the helper.** It's handed the parent's own cancel
object rather than a fresh one.

**The helper gets a hard turn cap.** Its own config could say twenty; it gets at
most `MAX_HELPER_TURNS`. A helper that loops is a bill the user didn't ask for.

The helper's run id is the parent's with a `/name` suffix, so its events are
visibly part of the parent's run while its receipt is still its own row. Anything
watching the stream should use `is_helper_run` to tell a helper finishing from the
whole conversation finishing.

`FanOutTool` is the parallel sibling: one helper mapped over a list of jobs and run
at once, under the same tool-parallelism ceiling an ordinary turn's tools use, then
gathered into one labelled answer. Each child is an ordinary helper run - same
session filing, same shared cancel, same per-child turn cap - distinguished only by
a `#i` suffix on its run id so its events and receipt stay attributable.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from ..config import AgentConfig
from ..conversation import Conversation, Session, system_message, user_message
from ..loop import Limits, RunContext, RunStatus
from .base import Tool, ToolDefinition, ToolPermissions, ToolResult

#: The most turns a helper may take, whatever its config says.
MAX_HELPER_TURNS = 6

#: What separates a helper's run id from its parent's.
HELPER_RUN_SEPARATOR = "/"

#: The name the model calls. Also the name removed from the helper's own tools.
TOOL_NAME = "delegate"

#: The name the model calls for the parallel version. Removed from a helper's own
#: tools alongside `TOOL_NAME`, so a helper can neither delegate nor fan out.
FANOUT_TOOL_NAME = "fan_out"

#: The most jobs one fan-out may open at once. A guard against a model asking for a
#: hundred helpers in a single call - each is a real run with a real bill. A wider
#: batch is refused with a message telling the model to split it, not silently
#: truncated. The per-run tool-parallelism ceiling still caps how many of these run
#: at the same time; this caps how many may be asked for at all.
MAX_FANOUT_WIDTH = 16


def is_helper_run(run_id: str) -> bool:
    """True if this run id belongs to a helper rather than a whole conversation.

    Worth having as a function: a UI watching the event stream stops when the run
    finishes, and without this it would stop at the first *helper* finishing and
    go quiet for the rest of the conversation.
    """
    return HELPER_RUN_SEPARATOR in (run_id or "")


class DelegateTool(Tool):
    """Runs a named helper on one job and returns only its answer."""

    def __init__(self, runtime: Any, max_turns: int = MAX_HELPER_TURNS) -> None:
        self._runtime = runtime
        self._max_turns = max_turns

    @property
    def definition(self) -> ToolDefinition:
        helpers = self._helper_names()
        schema = {
            "type": "object",
            "properties": {
                "helper": {
                    "type": "string",
                    "description": "Which helper to use."
                    + (f" One of: {', '.join(helpers)}." if helpers else ""),
                },
                "job": {
                    "type": "string",
                    "description": (
                        "The whole job, in words. The helper cannot see this "
                        "conversation, so say everything it needs - what to look "
                        "at, and what to tell you back."
                    ),
                },
            },
            "required": ["helper", "job"],
        }
        if helpers:
            schema["properties"]["helper"]["enum"] = helpers
        return ToolDefinition(
            name=TOOL_NAME,
            description=(
                "Hand a self-contained job to a helper and get back a short answer. "
                "Worth it when the work needs a lot of reading but you only need the "
                "conclusion - the reading stays out of this conversation. Not worth "
                "it for a single tool call you could just make yourself."
            ),
            input_schema=schema,
            # Not read-only: the helper can use tools of its own. Whatever it tries
            # goes through the same gate with the same policy, so a helper cannot
            # do anything the parent couldn't - it just has to be asked about too.
            permissions=ToolPermissions(read_only=False),
        )

    def preview(self, arguments: dict) -> str:
        job = str(arguments.get("job", ""))
        short = job if len(job) <= 80 else job[:77] + "..."
        return f'ask the {arguments.get("helper", "?")} helper to: {short}'

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        name = (arguments.get("helper") or "").strip()
        job = (arguments.get("job") or "").strip()
        if not job:
            return ToolResult(False, error="Nothing to delegate: `job` was empty.")

        config = self._config_for_helper(name, context)
        if config is None:
            known = ", ".join(self._helper_names()) or "none are configured"
            return ToolResult(False, error=f"No helper called {name!r}. Available: {known}.")

        result = await self._run_helper(config, job, context)

        if result.status == RunStatus.FINISHED:
            return ToolResult(True, output=result.final_text or "(the helper had nothing to say)")
        if result.status == RunStatus.LIMIT_REACHED:
            # Partial work is still work: hand back whatever it got to, and say so,
            # so the parent can decide whether to ask again more narrowly.
            return ToolResult(
                True,
                output=(
                    f"The {config.name} helper ran out of turns before finishing. "
                    f"What it had:\n{result.final_text or '(nothing yet)'}"
                ),
            )
        if result.status == RunStatus.CANCELLED:
            return ToolResult(False, error=f"The {config.name} helper was cancelled.")
        return ToolResult(False, error=f"The {config.name} helper failed: {result.error}")

    # -- the pieces ---------------------------------------------------------
    async def _run_helper(
        self, config: AgentConfig, job: str, context: Any, run_suffix: str = ""
    ) -> Any:
        """Give the helper its own conversation and run it to a stop.

        `run_suffix` sets the children of a single fan-out apart from one another:
        each gets `#0`, `#1`, ... appended to its run id, so its events and its
        receipt row stay its own. It defaults to empty, which leaves a plain
        `delegate` call's run id exactly `{parent}/{name}`, as it always was.
        """
        runtime = self._runtime
        parent_session = context.session

        # Where the helper's messages are filed. A real session row, so nothing is
        # orphaned and the helper's transcript can be read back later - just not by
        # the parent, which is the whole point.
        storage = Session(
            agent=config.name,
            title=f"helper for {parent_session.id}",
            working_directory=getattr(parent_session, "working_directory", "."),
        )
        await runtime.database.create_session(storage)

        tool_names = [t.definition.full_name for t in runtime.tools.get_available_tools(config)]
        conversation = Conversation(
            [
                system_message(runtime.prompt.build(config, storage, tool_names), parent_session.id),
                user_message(parent_session.id, job),
            ]
        )

        cap = self._helper_turn_cap(context)
        helper_context = RunContext(
            session=parent_session,  # so its events land on the parent's stream
            run_id=f"{context.run_id}{HELPER_RUN_SEPARATOR}{config.name}{run_suffix}",
            config=config,
            # A helper inherits the parent's operational envelope - the same retry
            # backoff and parallelism it was tuned with - so delegating doesn't
            # silently reset those. Only the turn ceiling is the helper's own.
            limits=Limits(
                max_turns=min(config.max_turns or cap, cap),
                max_retries=context.limits.max_retries,
                retry_first_delay_seconds=context.limits.retry_first_delay_seconds,
                max_parallel_tools=context.limits.max_parallel_tools,
                helper_max_turns=context.limits.helper_max_turns,
            ),
            cancellation=context.cancellation,  # stopping the parent stops the helper
        )
        return await self._helper_loop(storage.id).run(conversation, helper_context)

    def _helper_loop(self, storage_session_id: str) -> Any:
        """A loop exactly like the parent's, but filing messages elsewhere."""
        from ..loop import AgentLoop

        runtime = self._runtime
        return AgentLoop(
            runtime.models,
            runtime.tools,
            runtime.tool_manager,
            runtime.context,
            runtime.events,
            _HelperMessages(runtime.database, storage_session_id),
            runtime.monitoring,
            # The runtime's own coordinator, so a helper's retries back off in step
            # with the parent's rather than racing them into the same rate limit.
            retry_coordinator=getattr(runtime, "retry_coordinator", None),
            # ...and the runtime's hooks, so a pre/post-tool hook fires for a
            # helper's tools too, and a helper stopping fires SUBAGENT_STOP (the
            # loop tells the two apart by the helper separator in the run id).
            hooks=getattr(runtime, "hooks", None),
        )

    def _helper_turn_cap(self, context: Any) -> int:
        """The most turns a helper may take: the run's override, or this tool's default.

        `Limits.helper_max_turns` is 0 unless a caller set it on the run, in which
        case this tool's own default (`MAX_HELPER_TURNS`, set at construction)
        governs. That keeps the ceiling configurable per run without a magic number
        baked into the delegation path.
        """
        override = getattr(getattr(context, "limits", None), "helper_max_turns", 0) or 0
        return override or self._max_turns

    def _config_for_helper(self, name: str, context: Any) -> "AgentConfig | None":
        """Find the named helper, and narrow its tools so it can't delegate again.

        Two places to look: the `subagents` the calling agent declares, and the
        runtime's own agents. The declared ones win, because they're the ones this
        agent was told about.
        """
        if not name:
            return None

        parent = getattr(context, "config", None)
        found: Any = None
        for helper in getattr(parent, "subagents", None) or []:
            if getattr(helper, "name", "") == name:
                found = AgentConfig(
                    name=helper.name,
                    model=helper.model or getattr(parent, "model", ""),
                    description=helper.description,
                    system_prompt=helper.system_prompt or AgentConfig().system_prompt,
                    temperature=getattr(parent, "temperature", 0.0),
                    max_turns=self._helper_turn_cap(context),
                )
                break
        if found is None:
            registered = getattr(self._runtime, "agents", {}) or {}
            if name not in registered:
                return None
            found = registered[name]

        return replace(found, allowed_tools=self._tools_for_helper(found))

    def _tools_for_helper(self, config: Any) -> list:
        """The helper's tool list, always minus the two delegation tools.

        An empty `allowed_tools` means "all tools", so the removal has to be done
        by listing what's left rather than by subtracting from an empty list.
        """
        allowed = list(getattr(config, "allowed_tools", None) or [])
        if not allowed:
            allowed = [t.definition.full_name for t in self._runtime.tools.all()]
        # Neither delegation tool goes to a helper: no helper hires a helper, and no
        # helper opens a fan-out wave of its own. A tool a helper can't see is a
        # tool it can't call - no depth bookkeeping needed.
        barred = {TOOL_NAME, FANOUT_TOOL_NAME}
        return [name for name in allowed if name not in barred]

    def _helper_names(self) -> list:
        """Every helper the model could name, for the tool's own description."""
        names = set()
        for config in (getattr(self._runtime, "agents", {}) or {}).values():
            names.add(config.name)
            for helper in getattr(config, "subagents", None) or []:
                names.add(getattr(helper, "name", ""))
        return sorted(n for n in names if n)


class FanOutTool(DelegateTool):
    """Runs one helper across a list of jobs at once, and gathers every answer.

    The single-helper `DelegateTool` this extends already got the hard parts right -
    a helper's messages stay out of the parent's conversation, it can't delegate,
    it shares the parent's cancel, it has a hard turn cap. Fanning out reuses all of
    that unchanged; the only new thing is running several helpers at the same time
    instead of one after another.

    Three things a fan-out has to keep true, each inherited or arranged here:

    **The turn costs about the slowest child, not the sum.** The jobs run as
    concurrent tasks under `asyncio.gather`, capped by the same tool-parallelism
    ceiling an ordinary turn's tools run under, so five ten-second jobs finish in
    about ten seconds, not fifty.

    **One cancel stops all of them.** Every child is handed the parent's own cancel
    object (via `_run_helper`), so cancelling the parent makes each child stop at its
    next turn boundary rather than only the one currently talking to a model.

    **It can't multiply into a runaway bill.** Two guards: each child gets the same
    per-child turn cap a plain delegate does (`helper_max_turns`, else
    `MAX_HELPER_TURNS`), so N jobs cost at most N x cap turns, not N x the parent's
    own budget; and `MAX_FANOUT_WIDTH` refuses an over-wide wave outright rather than
    opening a hundred real runs from one call.

    Each child is an ordinary helper run distinguished by a `#i` suffix on its run
    id (`{parent}/{name}#0`, `#1`, ...), so its events on the parent's stream and its
    receipt row stay attributable to that job and no other.
    """

    @property
    def definition(self) -> ToolDefinition:
        helpers = self._helper_names()
        schema = {
            "type": "object",
            "properties": {
                "helper": {
                    "type": "string",
                    "description": "Which helper to run on every job."
                    + (f" One of: {', '.join(helpers)}." if helpers else ""),
                },
                "jobs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The jobs to run in parallel, one self-contained instruction "
                        "per item. Each helper sees only its own job and not this "
                        "conversation, so every item must say everything that job "
                        "needs - what to look at, and what to report back."
                    ),
                },
            },
            "required": ["helper", "jobs"],
        }
        if helpers:
            schema["properties"]["helper"]["enum"] = helpers
        return ToolDefinition(
            name=FANOUT_TOOL_NAME,
            description=(
                "Run one helper across several self-contained jobs at once and get "
                "back all the answers together. Worth it when the same kind of work "
                "has to be done over a list - many files to skim, many questions to "
                "check - and the jobs don't depend on each other. For a single job, "
                f"use {TOOL_NAME!r} instead."
            ),
            input_schema=schema,
            # Not read-only for the same reason `delegate` isn't: a child may use
            # tools of its own, each of which goes through the same gate and policy.
            permissions=ToolPermissions(read_only=False),
        )

    def preview(self, arguments: dict) -> str:
        jobs = arguments.get("jobs")
        count = len(jobs) if isinstance(jobs, list) else 0
        return f'fan the {arguments.get("helper", "?")} helper across {count} job(s)'

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        name = (arguments.get("helper") or "").strip()
        jobs = self._clean_jobs(arguments.get("jobs"))
        if isinstance(jobs, ToolResult):  # a shaped refusal, not a job list
            return jobs

        config = self._config_for_helper(name, context)
        if config is None:
            known = ", ".join(self._helper_names()) or "none are configured"
            return ToolResult(False, error=f"No helper called {name!r}. Available: {known}.")

        # Cap concurrency at the run's tool-parallelism ceiling, the same limit an
        # ordinary turn's tools share, so a wide fan-out can't open more model
        # connections at once than the rest of the loop is allowed to. `gather`
        # copies the current context into each child task, which keeps monitoring's
        # task-local run/span state isolated per child rather than tangled together.
        ceiling = max(1, getattr(context.limits, "max_parallel_tools", 1))
        limit = asyncio.Semaphore(ceiling)

        async def _one(index: int, job: str) -> Any:
            async with limit:
                return await self._run_helper(config, job, context, run_suffix=f"#{index}")

        results = await asyncio.gather(
            *(_one(index, job) for index, job in enumerate(jobs)),
            return_exceptions=True,  # one child crashing must not sink the others
        )
        return self._gather_results(config, jobs, results, context)

    # -- the pieces ---------------------------------------------------------
    def _clean_jobs(self, raw: Any) -> Any:
        """Validate the `jobs` list, or return a shaped `ToolResult` refusal.

        Returns a cleaned list of non-empty job strings on success. On any problem
        it returns a `ToolResult(False, ...)` whose message tells the model how to
        fix the call - an over-wide wave is refused with "split it", never silently
        truncated, so no job is dropped without the model knowing.
        """
        if not isinstance(raw, list):
            return ToolResult(False, error="`jobs` must be a list of job strings.")
        jobs = [str(job).strip() for job in raw if str(job).strip()]
        if not jobs:
            return ToolResult(False, error="Nothing to fan out: `jobs` was empty.")
        if len(jobs) > MAX_FANOUT_WIDTH:
            return ToolResult(
                False,
                error=(
                    f"Too many jobs at once ({len(jobs)}); the most is "
                    f"{MAX_FANOUT_WIDTH}. Split the list and fan out in batches."
                ),
            )
        return jobs

    def _gather_results(self, config: Any, jobs: list, results: list, context: Any) -> ToolResult:
        """Fold the children's outcomes into one labelled answer.

        A cancelled parent short-circuits to a single clear message rather than a
        wall of per-child cancellations. Otherwise every child gets a numbered line -
        its answer, or why it didn't have one - and the whole call succeeds if *any*
        child did, since partial results are still worth handing back.
        """
        cancel = getattr(context, "cancellation", None)
        if cancel is not None and getattr(cancel, "cancelled", False):
            return ToolResult(False, error=f"The {config.name} fan-out was cancelled.")

        blocks: list = []
        any_ok = False
        for index, result in enumerate(results):
            text, ok = self._child_result(config, result)
            any_ok = any_ok or ok
            blocks.append(f"[{index + 1}] {text}")
        joined = "\n\n".join(blocks)

        if any_ok:
            header = f"Fanned the {config.name} helper across {len(jobs)} jobs:\n\n"
            return ToolResult(True, output=header + joined)
        return ToolResult(False, error="Every child of the fan-out failed:\n\n" + joined)

    def _child_result(self, config: Any, result: Any) -> tuple:
        """One child's outcome as (text, ok). `ok` decides whether the whole call
        counts as having produced anything.

        A crash arrives as an exception (gather caught it for us); the run statuses
        mirror `DelegateTool.execute`'s single-helper handling - a finished helper's
        text, a limit-reached helper's partial work (still `ok`, as delegating one
        would be), and a clean not-ok for cancelled or errored.
        """
        if isinstance(result, BaseException):
            return (f"the {config.name} helper crashed: {result}", False)
        status = getattr(result, "status", None)
        if status == RunStatus.FINISHED:
            return (getattr(result, "final_text", "") or "(the helper had nothing to say)", True)
        if status == RunStatus.LIMIT_REACHED:
            partial = getattr(result, "final_text", "") or "(nothing yet)"
            return (f"(ran out of turns) {partial}", True)
        if status == RunStatus.CANCELLED:
            return (f"the {config.name} helper was cancelled.", False)
        return (f"the {config.name} helper failed: {getattr(result, 'error', '')}", False)


class _HelperMessages:
    """The database, with the helper's messages filed under a session of its own.

    Everything except `save_message` passes straight through, so the helper's run
    receipt, its events and any permission grant behave exactly as the parent's
    would. Only the messages move, and only so the parent's conversation stays the
    length the user expects.
    """

    def __init__(self, database: Any, session_id: str) -> None:
        self._db = database
        self._session_id = session_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    async def save_message(self, message: Any) -> None:
        message.session_id = self._session_id
        await self._db.save_message(message)

"""Lifecycle hooks: user callbacks at defined moments in a run.

A hook is a plain callable the user registers to run at a named point - before a
tool runs, after it returns, when a prompt is submitted, when a run or a subagent
stops. It's how someone bolts on auto-formatting, logging, or an extra policy gate
*without editing the agent*: the loop and the service call `HookManager.dispatch`
at each point, and whatever was registered runs there.

The event bus already announces these same moments to anything watching the
stream; hooks differ in two ways that make them worth having on top of it. They
run *in-band* - synchronously, in the run's own task - so a slow or blocking hook
actually holds the step it's attached to rather than racing it. And at the
pre-tool point a hook may **veto**: returning `HookOutcome(block=True, reason=...)`
stops the call from running, and the loop turns that into an ordinary refusal the
model reads and can react to. Every other point is observe-only; a return value
there is ignored.

Two properties the rest of the system leans on:

* **A hook must never crash a run.** A hook that raises is caught, its error
  recorded on the context, and dispatch carries on to the next hook - a bad
  logging callback cannot take down the agent. The one thing a hook can *do* on
  purpose is veto, and only at the pre-tool point.
* **No hooks means no change.** With nothing registered for a point, `dispatch`
  returns immediately and the call sites (guarded by `has`) do exactly what they
  did before hooks existed. "Disable the hooks and behaviour is identical" is a
  property, not a hope - which is what makes the feature safe to always wire in.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class HookPoint(str, Enum):
    """The moments a hook can attach to. String-valued so they log readably."""

    PRE_TOOL = "pre_tool"                 # before a tool runs; may veto
    POST_TOOL = "post_tool"               # after a tool returns; observe the result
    PROMPT_SUBMITTED = "prompt_submitted"  # a user message was accepted for a run
    RUN_STOP = "run_stop"                 # a top-level run reached a stopping point
    SUBAGENT_STOP = "subagent_stop"       # a helper (subagent) run stopped


@dataclass
class HookContext:
    """What a hook is told. Which fields are set depends on the point.

    `tool_name`/`arguments` are set at the tool points; `result` at POST_TOOL;
    `text` carries the user's prompt at PROMPT_SUBMITTED and the run's final text at
    the stop points; `status` names the run's outcome at the stop points. `errors`
    collects anything a hook raised, so a caller (or a test) can see a
    misbehaving hook without the run having noticed.

    At the tool points two more fields save a hook from having to look things up:
    `read_only` says whether the tool about to run (or that ran) only reads - so a
    hook that cares about mutations, like the auto-checkpointer, can skip a read in
    one comparison; and `working_directory` is the run's folder, so a hook that
    touches files knows where without reaching back into the session.
    """

    point: HookPoint
    session_id: str = ""
    run_id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    result: Any = None
    text: str = ""
    status: str = ""
    errors: list = field(default_factory=list)
    read_only: bool = False
    working_directory: str = ""


@dataclass
class HookOutcome:
    """A hook's optional say. Only `block` is acted on, and only at PRE_TOOL."""

    block: bool = False
    reason: str = ""


#: A hook takes the context and returns nothing, or a HookOutcome, or an awaitable
#: of either. Both sync and async callables work; dispatch awaits what needs it.
Hook = Callable[[HookContext], Any]


class HookManager:
    """Holds the registered hooks and runs them at each point, in order.

    Registration order is call order, because a chain of hooks (format, then lint,
    then log) usually has a meaning the user intends. The manager is deliberately
    tiny: no threads, no priorities, no event loop of its own - it's called from
    inside the run and does its work there.
    """

    def __init__(self) -> None:
        self._hooks: dict = {}

    def register(self, point: HookPoint, hook: Hook) -> None:
        """Add a hook at a point. Registering the same callable twice runs it twice."""
        self._hooks.setdefault(point, []).append(hook)

    def clear(self, point: "HookPoint | None" = None) -> None:
        """Remove hooks - at one point, or all of them. Restores hook-free behaviour."""
        if point is None:
            self._hooks.clear()
        else:
            self._hooks.pop(point, None)

    def has(self, point: HookPoint) -> bool:
        """Whether anything is registered at a point. The cheap guard call sites use
        so that, with no hooks, dispatch is never even entered."""
        return bool(self._hooks.get(point))

    async def dispatch(self, context: HookContext) -> "HookOutcome | None":
        """Run every hook at `context.point`, in order, and report a veto if one asks.

        Returns the first `HookOutcome(block=True, ...)` a hook produced - which only
        the PRE_TOOL call site acts on - or None. A hook that raises is caught, its
        error appended to `context.errors`, and the rest still run: one bad hook
        never stops the others and never ends the run.
        """
        hooks = self._hooks.get(context.point)
        if not hooks:
            return None
        blocking: "HookOutcome | None" = None
        for hook in hooks:
            try:
                outcome = hook(context)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
            except Exception as exc:  # noqa: BLE001 - a hook must not crash a run
                context.errors.append(
                    f"{getattr(hook, '__name__', repr(hook))}: {type(exc).__name__}: {exc}"
                )
                continue
            # Keep the first veto but keep running the rest, so observe-only hooks
            # registered after a vetoing one (a logger, say) still get to see the call.
            if blocking is None and isinstance(outcome, HookOutcome) and outcome.block:
                blocking = outcome
        return blocking

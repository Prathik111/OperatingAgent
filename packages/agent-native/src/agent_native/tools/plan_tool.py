"""A plan the model writes down - and nothing else.

One tool. It keeps a numbered list of steps with a status each, and every call
returns the whole list, so the current state of the plan is always the most recent
thing in the conversation. That's the entire feature.

**Nothing branches on it.** There is no executor, no scheduler, no step that gets
"run". The loop does not read this list and never will. That restraint is the
design, not an omission: v0.3 of this project failed because the plan drove
execution, and the fix isn't to stop planning - it's to make the plan something
the model reads rather than something that reads the model. A list in the
conversation helps a model still remember part four at turn twelve, and it can be
wrong without breaking anything.

**Why this is marked read-only.** The flag answers one question: can this change
something the user would still have after the conversation ends? A file, a
process, a network call, a note that outlives the session - those are all "no,
ask first". This list dies with the run and touches nothing outside it, so it
needs no prompt. (Compare `remember` in `memory_tools.py`, which does outlive the
session, and does ask.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Tool, ToolDefinition, ToolPermissions, ToolResult

TODO = "todo"
DOING = "doing"
DONE = "done"
STATUSES = (TODO, DOING, DONE)

#: A plan longer than this isn't a plan, it's a transcript.
MAX_STEPS = 20

#: One step should be a line, not a paragraph.
MAX_STEP_CHARS = 200


@dataclass
class Step:
    """One step, and whether it's been done."""

    text: str
    status: str = TODO


class PlanTool(Tool):
    """Keeps one list of steps per session and hands it back on every call."""

    def __init__(self) -> None:
        self._plans: dict = {}  # session id -> list[Step]

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="plan",
            description=(
                "Keep a short checklist for a job with several parts, so you don't "
                "lose track of it halfway. Pass `steps` to write or rewrite the "
                "list. Pass `step` and `status` to mark one as doing or done. Pass "
                "nothing to see the current list. Not worth it for a job that's one "
                "or two tool calls."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"The whole list, in order (up to {MAX_STEPS}). Replaces "
                            "any existing list; a step whose wording is unchanged "
                            "keeps the status it had."
                        ),
                    },
                    "step": {
                        "type": "integer",
                        "description": "Which step to update, counting from 1.",
                    },
                    "status": {
                        "type": "string",
                        "enum": list(STATUSES),
                        "description": "What that step is now: todo, doing, or done.",
                    },
                },
            },
            permissions=ToolPermissions(read_only=True),
        )

    def preview(self, arguments: dict) -> str:
        if arguments.get("steps"):
            return f"write a plan of {len(arguments['steps'])} steps"
        if arguments.get("step"):
            return f"mark step {arguments['step']} as {arguments.get('status', DONE)}"
        return "show the current plan"

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        session_id = _session_id_of(context)
        steps = self._plans.setdefault(session_id, [])
        notes: list = []

        if arguments.get("steps") is not None:
            steps = self._rewrite(session_id, arguments["steps"])
            if len(arguments["steps"]) > MAX_STEPS:
                notes.append(f"(kept the first {MAX_STEPS} steps)")

        number = arguments.get("step")
        if number is not None:
            ok, why = self._set_status(steps, number, arguments.get("status") or DONE)
            if not ok:
                # A bad step number is a result, not a failure: the model gets the
                # real list back and can see for itself what it should have said.
                return ToolResult(True, output=f"{why}\n\n{render(steps)}")

        if not steps:
            return ToolResult(True, output="No plan yet. Pass `steps` to write one.")
        return ToolResult(True, output=render(steps) + ("\n" + " ".join(notes) if notes else ""))

    # -- the pieces ---------------------------------------------------------
    def _rewrite(self, session_id: str, raw: Any) -> list:
        """Replace the list, keeping the status of any step worded the same as before.

        Rewriting is how a model revises a plan it got slightly wrong, and losing
        every tick each time it did so would make revising costly enough that it
        wouldn't - so the statuses are carried across by wording.
        """
        was = {step.text: step.status for step in self._plans.get(session_id, [])}
        steps = [
            Step(text=text, status=was.get(text, TODO))
            for text in (_clean(item) for item in (raw or []) if isinstance(item, str))
            if text
        ][:MAX_STEPS]
        self._plans[session_id] = steps
        return steps

    def _set_status(self, steps: list, number: Any, status: str) -> tuple:
        if not isinstance(number, int) or isinstance(number, bool):
            return False, f"`step` should be a whole number, not {number!r}."
        if not 1 <= number <= len(steps):
            return False, f"There is no step {number}; the plan has {len(steps)}."
        if status not in STATUSES:
            return False, f"`status` should be one of: {', '.join(STATUSES)}."
        steps[number - 1].status = status
        return True, ""

    def plan_for(self, session_id: str) -> list:
        """The current steps for a session. For a UI, or a test."""
        return list(self._plans.get(session_id, []))


def render(steps: list) -> str:
    """The plan as the model reads it back."""
    if not steps:
        return "(no plan)"
    lines = [f"{i}. [{step.status}] {step.text}" for i, step in enumerate(steps, 1)]
    left = sum(1 for step in steps if step.status != DONE)
    lines.append(f"({len(steps) - left} of {len(steps)} done)")
    return "\n".join(lines)


def _clean(text: str) -> str:
    text = " ".join(text.split())
    return text[:MAX_STEP_CHARS] if len(text) > MAX_STEP_CHARS else text


def _session_id_of(context: Any) -> str:
    session = getattr(context, "session", None)
    return getattr(session, "id", "") if session is not None else ""

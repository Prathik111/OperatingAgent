"""Invoke a skill: load its full instructions, only when the model asks.

The catalogue of skills sits in the system prompt as one line each (see
`config.discover_skills` / `config.skill_listing`). This tool is the second half
of the progressive-disclosure pattern: given a skill's name, it reads that
skill's `SKILL.md` body fresh from disk and hands it back, so the full
instructions enter the conversation only when a task actually calls for them.

**Read-only, like `plan` and unlike `remember`.** Loading instructions changes
nothing that outlives the run, so the policy never stops to ask. The name is
resolved against the skills discovered under the working folder - it is never
used as a path - so there is no way to point this outside the project, and it
needs no path fence of its own.

**Re-discovers on each call, on purpose.** The prompt's catalogue is a snapshot
from when the session began; the body is read live here. A skill edited mid-session
is therefore picked up on the next invoke, and the tool holds no state of its own -
the same reasoning that has `read_project_instructions` re-read `AGENT.md`.
"""

from __future__ import annotations

from typing import Any

from ..config import discover_skills
from .base import Tool, ToolDefinition, ToolPermissions, ToolResult

#: A single skill body long enough to swamp the conversation is trimmed; the model
#: still gets the opening instructions, which is where a SKILL.md front-loads them.
MAX_SKILL_BODY_CHARS = 20_000


class InvokeSkillTool(Tool):
    """Hands back one named skill's full instructions, read live from disk."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="invoke_skill",
            description=(
                "Load the full instructions for one of the skills listed in your "
                "system prompt. Call this the moment a task matches a skill - before "
                "doing the work - and then follow the instructions it returns. Pass "
                "the skill's `name`; an unknown name lists what's available."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load, as listed in the prompt.",
                    }
                },
                "required": ["name"],
            },
            permissions=ToolPermissions(read_only=True),
        )

    def preview(self, arguments: dict) -> str:
        return f"load skill {(arguments.get('name') or '').strip()!r}"

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        skills = discover_skills(_working_directory_of(context))
        name = (arguments.get("name") or "").strip()
        if not name:
            return ToolResult(True, output="Which skill? " + _catalogue(skills))

        # A missing skill is a result the model can recover from, not a failure:
        # it gets the real list back and can pick a name that exists (the same way
        # plan_tool answers a bad step number).
        match = next((skill for skill in skills if skill.name.lower() == name.lower()), None)
        if match is None:
            return ToolResult(True, output=f"No skill named {name!r}. " + _catalogue(skills))

        body = match.body()
        if not body:
            return ToolResult(True, output=f"Skill {match.name!r} has no readable instructions.")
        truncated = len(body) > MAX_SKILL_BODY_CHARS
        if truncated:
            body = body[:MAX_SKILL_BODY_CHARS] + "\n... [skill truncated]"
        return ToolResult(True, output=f"Skill: {match.name}\n\n{body}", truncated=truncated)


def _catalogue(skills: list) -> str:
    """The available skill names, for when the model asked for one that isn't there."""
    if not skills:
        return "No skills are available in this workspace."
    return "Available skills: " + ", ".join(skill.name for skill in skills) + "."


def _working_directory_of(context: Any) -> str:
    session = getattr(context, "session", None)
    return (getattr(session, "working_directory", "") or "") if session is not None else ""

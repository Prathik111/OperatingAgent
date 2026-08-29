"""Two tools: keep a note, and look one up.

These are the only tools that ship inside the agent itself - everything else
comes from the MCP gateway. They're here rather than out there because a note is
part of the agent's own bookkeeping, not a service it borrows.

**Why `remember` asks the user.** Its flags are honest: it isn't read-only, so
the policy floor stops and asks before the note is stored. That looks like
friction and is actually the point. A memory system that quietly accumulates
notes about someone is the thing people complain about after the fact, and the
prompt shows the exact sentence that is about to be kept, so the answer is easy.
Notes are rare - a handful in a long session - so the cost is small. `recall`
only reads, so it never interrupts.
"""

from __future__ import annotations

from typing import Any

from ..memory import VALID_KINDS, MemoryKind, MemoryStore
from .base import Tool, ToolDefinition, ToolPermissions, ToolResult


class RememberTool(Tool):
    """Write down something worth knowing next time."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="remember",
            description=(
                "Keep a short note that will still be available in future "
                "conversations. Use it for a lasting preference (\"always run tests "
                "with uv\"), a fact about the project, or a correction the user just "
                "made. Don't use it for anything only relevant to right now."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note, in one or two plain sentences.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": sorted(VALID_KINDS),
                        "description": (
                            "preference (how the user likes things done), fact "
                            "(something true about the project), or correction "
                            "(something you got wrong)."
                        ),
                    },
                },
                "required": ["text"],
            },
            permissions=ToolPermissions(read_only=False),
        )

    def preview(self, arguments: dict) -> str:
        """Show the note itself - that's the whole thing being approved."""
        kind = arguments.get("kind") or MemoryKind.FACT
        return f'remember this {kind}: "{arguments.get("text", "")}"'

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        text = (arguments.get("text") or "").strip()
        if not text:
            return ToolResult(False, error="Nothing to remember: `text` was empty.")
        memory = await self._store.remember(
            text,
            kind=arguments.get("kind") or MemoryKind.FACT,
            session_id=_session_id_of(context),
        )
        return ToolResult(True, output=f"Remembered ({memory.kind}): {memory.text}")


class RecallTool(Tool):
    """Look up notes kept earlier."""

    def __init__(self, store: MemoryStore, limit: int = 5) -> None:
        self._store = store
        self._limit = limit

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="recall",
            description=(
                "Look up notes you kept earlier, by keyword. Worth doing before "
                "assuming how something should be done, in case the user already "
                "said."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Words to look for, e.g. \"tests\" or \"deploy\".",
                    }
                },
                "required": ["query"],
            },
            permissions=ToolPermissions(read_only=True),
        )

    def preview(self, arguments: dict) -> str:
        return f'recall notes about "{arguments.get("query", "")}"'

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        query = (arguments.get("query") or "").strip()
        found = await self._store.recall(
            query, session_id=_session_id_of(context), limit=self._limit
        )
        if not found:
            # A miss is a success with an empty answer, not an error: "nothing is
            # written down about this" is a useful, true thing for the model to read.
            return ToolResult(True, output=f"No notes about {query!r}.")
        return ToolResult(True, output=format_memories(found))


def memory_tools(store: MemoryStore) -> list:
    """Both tools, ready to register."""
    return [RememberTool(store), RecallTool(store)]


def format_memories(memories: list) -> str:
    """Notes as lines the model reads, in both the tool result and the first prompt."""
    return "\n".join(f"- ({m.kind}) {m.text}" for m in memories)


def _session_id_of(context: Any) -> str:
    session = getattr(context, "session", None)
    return getattr(session, "id", "") if session is not None else ""

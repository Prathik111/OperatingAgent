"""What a tool is, and the book of tools.

A tool is anything the model can ask to run: read a file, run a command, search.
Each one carries a `ToolDefinition` - its name, what arguments it takes, and a
few honest flags about what it does (is it read-only? could it destroy something?
does it need the network? where should it run?). Those flags are what the policy
reads when it decides whether to just allow the call or stop and ask.

`ExecutionMode` is the one that matters most for safety: DIRECT runs inside the
agent, HOST_PROCESS runs in a child process with the user's full privileges, and
SANDBOX runs locked in a container. The MCP tools (reached in-memory) run DIRECT,
except a shell command, which is marked SANDBOX - so it runs in a container when
the machine has Docker, and in a child process with the old allowlist when it
doesn't. Either way the policy always asks first: a container stops a command from
reaching the rest of the machine, not from wrecking the user's own project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    """Where a tool runs, from most trusted to least."""

    DIRECT = "direct"              # inside the agent process
    HOST_PROCESS = "host_process"  # a child process, full user privileges
    SANDBOX = "sandbox"            # a locked-down container


@dataclass
class ToolPermissions:
    """Honest flags about what a tool does. The policy reads these.

    `read_only` answers one question: could this change anything the user would
    still have after the conversation ends? A file, a process, a network call, a
    note that outlives the session - all of those are "no", and the policy stops to
    ask. A scratchpad that dies with the run is not.

    It is a claim about EFFECTS, not about REACH: a read-only tool cannot change
    anything, but it may still be able to *look* anywhere on the disk. That is what
    `reaches_paths` is for - it marks a tool whose reach is decided by a path
    argument the model supplies, so the policy can check where that path actually
    lands instead of trusting "it only reads".
    """

    read_only: bool = False
    destructive: bool = False
    needs_network: bool = False
    reaches_paths: bool = False
    execution_mode: ExecutionMode = ExecutionMode.DIRECT


@dataclass
class ToolDefinition:
    """Everything about a tool except how it runs."""

    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)
    permissions: ToolPermissions = field(default_factory=ToolPermissions)
    namespace: str = ""
    #: How long this one call may take. 0 means "use the shared default" in
    #: ToolManager. Set it on a tool that is legitimately slow (a build, a big
    #: search) rather than raising the default for everything.
    timeout_seconds: float = 0.0

    @property
    def full_name(self) -> str:
        return f"{self.namespace}.{self.name}" if self.namespace else self.name


@dataclass
class ToolResult:
    """What a tool hands back. Success or failure - both are just data."""

    success: bool
    output: str = ""
    error: str = ""
    truncated: bool = False


class Tool(ABC):
    """The one thing every tool must be: a definition plus a way to run."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    async def execute(self, arguments: dict, context: Any) -> ToolResult: ...

    def preview(self, arguments: dict) -> str:
        """A one-line, human-readable summary of what this call will do.

        Shown in the permission prompt, so the user sees the real operation - not
        a planner's guess. Destructive tools should override this to be specific.
        """
        return f"{self.definition.full_name}({arguments})"

    def sandbox_command(self, arguments: dict) -> str | list | None:
        """The shell command that does this call's work, if it is one.

        A tool marked `ExecutionMode.SANDBOX` is asked this before it is run: if it
        answers with a command, that command runs inside the session's container
        instead of `execute` running here (see `tools/sandbox.py`). Answering None -
        the default - means "I can't be expressed as a command", and the tool is
        run normally. A string is handed to a shell inside the container; a list is
        executed directly.
        """
        return None



class ToolRegistry:
    """The book of every tool the agent knows about."""

    def __init__(self) -> None:
        self._tools: dict = {}  # full_name -> Tool

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.full_name] = tool

    def find(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list:
        return list(self._tools.values())

    def get_available_tools(self, agent: Any) -> list:
        """The tools this agent is allowed to use. Empty allow-list means all."""
        allowed = getattr(agent, "allowed_tools", None)
        if not allowed:
            return self.all()
        return [t for t in self._tools.values() if t.definition.full_name in allowed]

    def native_schemas(self, agent: Any) -> list:
        """The tool list in the shape a native-tool-calling model expects."""
        return [native_schema(t.definition) for t in self.get_available_tools(agent)]


def native_schema(definition: ToolDefinition) -> dict:
    """One tool, in the OpenAI / Groq function-tool shape."""
    return {
        "type": "function",
        "function": {
            "name": definition.full_name,
            "description": definition.description,
            "parameters": definition.input_schema or {"type": "object", "properties": {}},
        },
    }


class ArgumentChecker:
    """Checks a tool call's arguments against its schema before anything runs.

    Deliberately small - it catches the mistakes that matter (a missing required
    argument, a string where a number belongs) and turns them into a plain reason
    string. A full JSON-Schema validator is more than the loop needs.
    """

    def validate(self, schema: dict, arguments: dict) -> tuple:
        """Return (ok, reason). reason is empty when ok is True."""
        if not schema:
            return True, ""
        if schema.get("type") == "object":
            props = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in arguments:
                    return False, f"missing required argument: {key}"
            for key, value in arguments.items():
                if key in props:
                    expected = props[key].get("type")
                    if expected and not _type_ok(expected, value):
                        return False, f"argument {key!r} should be {expected}"
        return True, ""


def _type_ok(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True  # unknown type: don't block

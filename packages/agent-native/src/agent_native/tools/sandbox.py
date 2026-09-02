"""Running a tool inside a container instead of on the machine.

This is the seam between the agent and `packages/sandbox`. `ToolManager` holds one
of these and asks it to run any tool marked `ExecutionMode.SANDBOX`; everything
else it runs itself, as before. That single branch is the whole integration, and it
only exists in one place because every tool call already goes through one gate.

**A tool that can be sandboxed says so by having a command.** `Tool.sandbox_command`
returns the shell command that does its work, or None. A shell tool obviously has
one. A tool that pokes at the agent's own state does not, and returns None - which
is also the answer for "this tool can't be moved into a container", so nothing has
to be listed anywhere.

**`None` back from `run` means "I couldn't", not "it failed".** No Docker, no
working folder, a tool with no command: all of them mean the manager should carry
on and run the tool directly. A machine without Docker is not a machine where the
agent refuses to work.

Two decisions the plan asked to be made out loud rather than discovered:

**The file tools stay on the host.** They are not shell commands - they go to the
MCP file server, which already refuses any path outside the working folder, and
moving them into the container would mean shelling out for every read and losing
that check. The cost is real and worth saying: the shell sees the project at
`/workspace` while the file tools see it at its actual path. What makes that
liveable is that `/workspace` is the container's working directory, so *relative*
paths mean the same thing to both, and relative paths are what a model uses when
the conversation is about a project folder. An absolute host path pasted into a
shell command won't resolve, and the error will say so.

**The allowlist isn't relaxed - it's bypassed.** Inside the container there is no
allowlist, because the container is doing that job properly: `rm -rf /` there
destroys a container we throw away. On the fallback path (no Docker) the command
still goes through the MCP terminal server, and its conservative allowlist still
applies - because on that path it is the only thing standing there. Relaxing the
list would have loosened exactly the case that has no container to protect it.
"""

from __future__ import annotations

import os
from typing import Any

from .base import ExecutionMode, Tool, ToolResult

#: The sandbox stops a command slightly before the manager's own deadline, so the
#: model reads the specific message ("timed out in the sandbox") rather than the
#: generic one, and so the container is still there to be asked what happened.
TIMEOUT_MARGIN_SECONDS = 5.0


class ContainerSandbox:
    """Runs sandbox-marked tools in a per-session container, or admits it can't."""

    def __init__(
        self,
        image: str = "",
        memory: str = "",
        cpus: str = "",
        network: bool = False,
    ) -> None:
        self._image = image
        self._memory = memory
        self._cpus = cpus
        self._network = network
        self._pool: Any = None
        self._reason = ""      # why containers aren't being used, if they aren't
        self._used = False     # did anything actually run in a container?

    # -- what the manager calls ---------------------------------------------
    async def run(
        self,
        tool: Tool,
        arguments: dict,
        context: Any,
        timeout: float,
    ) -> ToolResult | None:
        """Run this tool in the session's container. None means "couldn't"."""
        command = tool.sandbox_command(dict(arguments or {}))
        if not command:
            self._note(f"{tool.definition.full_name} has no command to run in a container")
            return None

        root = _working_directory(context)
        if not root:
            self._note("no working folder for this session")
            return None

        pool = self._get_pool()
        if pool is None:
            return None

        runner = await pool.get(_session_id_of(context), root)
        if runner is None:
            self._note(pool.reason or "couldn't start a container")
            return None

        seconds = max(1.0, timeout - TIMEOUT_MARGIN_SECONDS)
        output = await runner.run(command, timeout=seconds)
        self._used = True
        return _as_tool_result(output, seconds)

    def handles(self, tool: Tool) -> bool:
        """Whether a call to this tool would be routed here at all.

        The manager uses this to tell the user, in the permission prompt, where the
        thing they're approving is about to run. It deliberately doesn't start
        anything or touch Docker - a prompt shouldn't wait a second on a probe -
        so it answers from the tool's mark alone, and can be wrong in the safe
        direction: if the container turns out to be unavailable, the call falls back
        and the startup line already said the sandbox was off.
        """
        return tool.definition.permissions.execution_mode is ExecutionMode.SANDBOX

    # -- for a startup banner, and for the run receipt -----------------------
    async def probe(self) -> bool:
        """Check Docker now rather than on the first command. Never raises.

        The reason is kept when this fails, because the line printed at startup is
        the only place a user finds out they aren't sandboxed - and "off" without a
        reason is a worse message than no message.
        """
        pool = self._get_pool()
        if pool is None:
            return False
        try:
            if await pool.available():
                return True
        except Exception as exc:
            self._note(f"{exc}" or getattr(pool, "reason", "") or "no container available")
            return False
        self._note(getattr(pool, "reason", "") or "no container available")
        return False

    def status_line(self) -> str:
        """One line saying which mode this run is in, and why.

        The plan asked for this explicitly, and it earns its place: "the agent ran
        my command in a container" and "the agent ran my command on my laptop" are
        different enough that a user should never have to guess which happened.
        """
        pool = self._pool
        if pool is None:
            return f"sandbox: off - {self._reason or 'not enabled'}"
        if self._reason and not self._used:
            return (
                f"sandbox: off - {self._reason}. Shell commands run on your machine, "
                "limited to the allowlist and the working folder."
            )
        return f"sandbox: on - shell commands run in a container ({pool.image}, no network)"

    async def close(self) -> None:
        """Remove every container this run started."""
        if self._pool is not None:
            await self._pool.stop_all()

    # -- the lazy import, in one place --------------------------------------
    def _get_pool(self) -> Any:
        """Build the pool on first use. Missing package is a reason, not a crash."""
        if self._pool is not None:
            return self._pool
        try:
            import sandbox as sandbox_package
        except ImportError as exc:
            self._note(
                f"the 'sandbox' package isn't installed ({exc}); "
                "from the repo root run `uv sync --all-packages`"
            )
            return None
        required = ("ContainerPool", "DEFAULT_CPUS", "DEFAULT_IMAGE", "DEFAULT_MEMORY")
        missing = [name for name in required if not hasattr(sandbox_package, name)]
        if missing:
            self._note(
                "the installed 'sandbox' package does not provide the container "
                f"runtime ({', '.join(missing)} missing)"
            )
            return None
        exports = vars(sandbox_package)
        ContainerPool = exports["ContainerPool"]
        DEFAULT_CPUS = exports["DEFAULT_CPUS"]
        DEFAULT_IMAGE = exports["DEFAULT_IMAGE"]
        DEFAULT_MEMORY = exports["DEFAULT_MEMORY"]
        self._pool = ContainerPool(
            image=self._image or DEFAULT_IMAGE,
            memory=self._memory or DEFAULT_MEMORY,
            cpus=self._cpus or DEFAULT_CPUS,
            network=self._network,
        )
        return self._pool

    def _note(self, reason: str) -> None:
        self._reason = reason


def _as_tool_result(output: Any, seconds: float) -> ToolResult:
    """Turn a container command's outcome into a result the model can read.

    A non-zero exit puts everything in `error`, output included, because that is
    the field the conversation shows for a failed call - and the output of a failed
    command is usually the part that explains it.
    """
    text = output.combined()
    if output.timed_out:
        return ToolResult(
            False,
            error=(
                f"Timed out after {seconds:g}s in the sandbox container."
                + (f" Output so far:\n{text}" if text else "")
            ),
        )
    if output.exit_code != 0:
        return ToolResult(
            False,
            error=f"Command failed (exit {output.exit_code}) in the sandbox container."
            + (f"\n{text}" if text else ""),
        )
    return ToolResult(True, output=text or "(no output)")


def _working_directory(context: Any) -> str:
    session = getattr(context, "session", None)
    root = getattr(session, "working_directory", "") if session is not None else ""
    if not root:
        return ""
    path = os.path.abspath(os.path.expanduser(root))
    return path if os.path.isdir(path) else ""


def _session_id_of(context: Any) -> str:
    session = getattr(context, "session", None)
    return getattr(session, "id", "") if session is not None else ""

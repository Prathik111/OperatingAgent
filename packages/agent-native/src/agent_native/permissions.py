"""Who is allowed to do what.

Two layers, kept apart on purpose:

  - a POLICY looks at a tool call and gives a verdict: allow it, ask the user, or
    deny it outright. Policies are pure - same call, same verdict - so they are
    easy to reason about and test. A `RulePolicy` reads the tool's own flags
    (read-only calls are allowed, anything that can destroy is asked about) plus
    any explicit rules; a `WorkspacePolicy` checks where the call's path arguments
    actually land, which is the part the flags cannot express; a `SessionPolicy`
    adds memory ("you've reached a new site, I'll ask the first time"). A
    `PolicyChain` runs several and takes the strictest answer, so denial always
    wins.

  - the PERMISSION MANAGER is what actually asks the user. When a policy says
    "ask", it raises a request on the event bus (with the real tool name, the
    real arguments and a preview), then waits for the answer to come back. If the
    user says "allow for the session", it remembers, so they aren't asked twice -
    and that yes can be narrowed to a place, so "writes under `notes/` are fine"
    is something they can say instead of "writes are fine".

The important property: a policy can hide nothing. The prompt the user sees is
built from the same arguments the tool will actually run with.
"""

from __future__ import annotations

import asyncio
import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .events import EventType


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Strictness order, so a chain can pick the most restrictive verdict.
_STRICTNESS = {PermissionDecision.ALLOW: 0, PermissionDecision.ASK: 1, PermissionDecision.DENY: 2}


@dataclass
class Decision:
    """A verdict plus why - the reason is what the user or a log will read."""

    result: PermissionDecision
    reason: str = ""
    rule: str = ""


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class Policy(ABC):
    """Looks at one tool call and returns a verdict."""

    @abstractmethod
    def check(self, context: Any, definition: Any, arguments: dict) -> Decision: ...


@dataclass
class PermissionRule:
    """An explicit override: 'calls to git.* always ask', 'filesystem_read_file always allow'."""

    id: str
    tool_pattern: str
    result: PermissionDecision
    argument_pattern: str = ""  # optional substring that must appear in the arguments

    def matches(self, tool_name: str, arguments: dict) -> bool:
        if not fnmatch.fnmatch(tool_name, self.tool_pattern):
            return False
        return not self.argument_pattern or self.argument_pattern in str(arguments)


class RulePolicy(Policy):
    """The tool's own honest flags set a floor; explicit rules may only tighten it.

    The floor is deny-by-default: read-only calls are allowed, everything else is
    asked about. An explicit rule can make a call *stricter* (ask -> deny), but it
    can never relax below the floor - only a user grant does that. This is what
    stops a stray 'allow' rule from quietly waving through a destructive tool.
    """

    def __init__(self, rules: list | None = None) -> None:
        self.rules: list = list(rules or [])

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        name = definition.full_name
        floor = self._capability_floor(definition)

        for rule in self.rules:
            if rule.matches(name, arguments):
                verdict = Decision(rule.result, reason=f"rule {rule.id}", rule=rule.id)
                # A rule may tighten, never loosen: keep whichever is stricter.
                if _STRICTNESS[verdict.result] >= _STRICTNESS[floor.result]:
                    return verdict
                return floor

        return floor

    def _capability_floor(self, definition: Any) -> Decision:
        """The least-restrictive verdict the tool's own flags permit."""
        perms = definition.permissions
        if perms.destructive:
            return Decision(PermissionDecision.ASK, reason="this can change or delete files")
        if perms.needs_network:
            return Decision(PermissionDecision.ASK, reason="this reaches the network")
        if perms.read_only:
            # Read-only is a claim about effects, not reach. Where a read-only tool
            # is allowed to *look* is WorkspacePolicy's job, not this floor's.
            return Decision(PermissionDecision.ALLOW, reason="read-only")
        # Not read-only and not obviously safe: ask rather than assume.
        return Decision(PermissionDecision.ASK, reason="not a read-only action; asking to be safe")


# Argument names that carry a filesystem location. Matched as substrings, so
# `path`, `file_path`, `source_dir` and `repository` are all caught.
_PATH_ARGUMENT_HINTS = (
    "path",
    "dir",
    "file",
    "repo",
    "source",
    "destination",
    "target",
    "cwd",
)


class WorkspacePolicy(Policy):
    """Asks when a tool's path argument points outside the workspace folder.

    This is the answer to the hole that `read_only` leaves open. A read-only tool
    cannot change anything, but nothing in its flags says where it may *look* - so
    a repository argument of `/etc` on an otherwise honest read tool would sail
    through the floor. This policy reads the arguments the tool will really run
    with and objects when one of them leaves the folder the user opened.

    The folder comes from the session (`working_directory`), so two sessions on
    two projects get two different fences without anything being re-wired. Pass
    `root` to pin one explicitly, which is what the tests do.

    Relative paths are treated as inside: the MCP servers resolve them against
    their own confined root, so `config.txt` can only ever mean a file in the
    workspace. Absolute paths are resolved and checked for containment.

    With no folder to compare against, the answer is always "ask". That is
    deliberate - an unbounded reach is exactly the case worth stopping for, so the
    default when nobody has said where the workspace is must be the cautious one.
    """

    def __init__(self, root: str | None = None) -> None:
        self._pinned_root = root

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        if not getattr(definition.permissions, "reaches_paths", False):
            return Decision(PermissionDecision.ALLOW, reason="does not take a path")

        root = self._root_for(context)
        if root is None:
            return Decision(
                PermissionDecision.ASK,
                reason="this takes a path and no workspace folder is set, so its reach is unbounded",
            )

        for value in _path_arguments(arguments):
            outside = _outside_root(value, root)
            if outside is not None:
                return Decision(
                    PermissionDecision.ASK,
                    reason=f"{outside} is outside the workspace folder ({root})",
                )
        return Decision(PermissionDecision.ALLOW, reason="inside the workspace folder")

    def _root_for(self, context: Any) -> Path | None:
        """The fence for this call: the pinned root, else the session's folder."""
        raw = self._pinned_root
        if raw is None:
            session = getattr(context, "session", None)
            raw = getattr(session, "working_directory", None) if session is not None else None
        if not raw:
            return None
        try:
            return Path(raw).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return None


def _outside_root(value: str, root: Path) -> str | None:
    """The path, if it escapes the root; None if it stays inside or is relative."""
    try:
        candidate = Path(value).expanduser()
    except (TypeError, ValueError):
        return None
    if not candidate.is_absolute():
        # The servers resolve relative paths against their own root, so a relative
        # path cannot leave the workspace. A `..` inside one is caught there.
        return None
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return value
    if resolved == root or root in resolved.parents:
        return None
    return str(resolved)


def _path_arguments(arguments: dict) -> list:
    """Every argument value that looks like a filesystem location."""
    found: list = []
    for key, value in (arguments or {}).items():
        if not any(hint in key.lower() for hint in _PATH_ARGUMENT_HINTS):
            continue
        if isinstance(value, str) and value:
            found.append(value)
        elif isinstance(value, (list, tuple)):
            found.extend(item for item in value if isinstance(item, str) and item)
    return found


class SessionPolicy(Policy):
    """Adds memory within a single session: the first time something new happens, ask.

    Memory is kept per session, so reaching a host in one session never silently
    approves it in another. A missing context (as in unit tests) falls into a
    shared '' bucket.
    """

    def __init__(self) -> None:
        self._seen_hosts: dict = {}  # session_id -> set of hosts already seen

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        perms = definition.permissions
        if perms.needs_network:
            session_id = _session_id_of(context)
            seen = self._seen_hosts.setdefault(session_id, set())
            host = str(arguments.get("host") or arguments.get("url") or "")
            if host and host not in seen:
                seen.add(host)
                return Decision(PermissionDecision.ASK, reason=f"first time reaching {host}")
        # No opinion otherwise. ALLOW here means "I don't object"; the chain
        # still lets a stricter policy ask or deny.
        return Decision(PermissionDecision.ALLOW, reason="no session concern")


def _session_id_of(context: Any) -> str:
    """Pull a session id out of a run context, tolerating None (tests pass None)."""
    session = getattr(context, "session", None)
    return getattr(session, "id", "") if session is not None else ""


class PlanModePolicy(Policy):
    """The hard half of plan mode: while it's on, only read-only calls may run.

    Plan mode confines a run to investigating and proposing - no writing a file,
    running a command, or leaving a note that outlives the session - until the user
    has seen the plan and approved it. The loop already narrows the *visible* tools
    to read-only ones in plan mode (see `AgentLoop._tool_schemas`), so a well-behaved
    model never reaches for a mutating tool; this policy is what turns that from a
    hope into a guarantee. A mutating call the model makes anyway - a name it
    hallucinated, a tool it saw in an earlier non-plan turn - is denied here, and
    denial wins in the chain.

    Off (the default, and any run that never set the flag) it has no opinion, so
    adding it to the chain changes nothing for an ordinary run. "Approval" isn't a
    state this policy tracks: it's simply the next run made without plan mode, where
    full tools are visible and this policy stays silent.

    It reads the flag off the run context defensively - a missing context or missing
    limits (as some unit tests pass) reads as "not in plan mode", the safe default
    that leaves the call to the other policies.
    """

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        limits = getattr(context, "limits", None)
        if not getattr(limits, "plan_mode", False):
            return Decision(PermissionDecision.ALLOW, reason="not in plan mode")
        if getattr(definition.permissions, "read_only", False):
            return Decision(PermissionDecision.ALLOW, reason="read-only, allowed while planning")
        return Decision(
            PermissionDecision.DENY,
            reason="plan mode is on: only read-only tools may run until the plan is approved",
        )


class PolicyChain(Policy):
    """Runs several policies and returns the strictest verdict (deny > ask > allow)."""

    def __init__(self, policies: list) -> None:
        self.policies: list = list(policies)

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        strictest = Decision(PermissionDecision.ALLOW, reason="nothing objected")
        for policy in self.policies:
            decision = policy.check(context, definition, arguments)
            if _STRICTNESS[decision.result] > _STRICTNESS[strictest.result]:
                strictest = decision
        return strictest


# ---------------------------------------------------------------------------
# Grants: remembering the user's answer
# ---------------------------------------------------------------------------
class PermissionDuration(str, Enum):
    ONCE = "once"        # just this call
    SESSION = "session"  # the rest of this session
    ALWAYS = "always"    # forever, across sessions


@dataclass
class PermissionGrant:
    """A remembered 'yes' for tools matching a pattern, optionally within a path.

    Without `argument_pattern` a grant is all-or-nothing: "yes, write files for the
    rest of this session" covers writing *anywhere* the policy would otherwise ask
    about, which is more than most users mean when they say yes. The pattern is
    how they say the smaller thing - "writes under `notes/` are fine" - and it is
    the difference between a grant a careful person will use and one they won't.
    """

    tool_pattern: str
    duration: PermissionDuration
    session_id: str = ""  # empty means it applies everywhere (ALWAYS)
    argument_pattern: str = ""  # empty means "any arguments"

    def covers(self, tool_name: str, session_id: str, arguments: dict | None = None) -> bool:
        if not fnmatch.fnmatch(tool_name, self.tool_pattern):
            return False
        if self.session_id and self.session_id != session_id:
            return False
        if not self.argument_pattern:
            return True
        return self._covers_paths(arguments or {})

    def _covers_paths(self, arguments: dict) -> bool:
        """Whether *every* path this call touches is inside what was approved.

        Every, not any. A call that writes to `notes/x.md` and `/etc/hosts` is not
        covered by "notes/ is fine" - matching on the first path found would turn a
        narrow yes into a wide one, which is the exact failure this feature exists
        to avoid. A call with no path at all isn't covered either: the user scoped
        their answer to a place, and a call that names no place isn't in it.

        Not covered means the user is asked again - so being wrong here costs a
        prompt, never an unapproved write.
        """
        paths = _path_arguments(arguments)
        if not paths:
            return False
        return all(_within_pattern(value, self.argument_pattern) for value in paths)


def _within_pattern(value: str, pattern: str) -> bool:
    """Whether one path argument falls inside a grant's pattern.

    A pattern with wildcards in it is a glob, matched as written. A pattern
    without them is read as a folder prefix, because that's what a person means by
    "notes/" - the folder and everything under it, not a file with that exact
    name. Separators are normalised so a Windows-shaped argument matches a
    pattern typed with forward slashes.
    """
    text = _normalise(value)
    wanted = _normalise(pattern)
    if not text or not wanted:
        return False
    if any(char in wanted for char in "*?["):
        return fnmatch.fnmatch(text, wanted)
    wanted = wanted.rstrip("/")
    return text == wanted or text.startswith(wanted + "/")


def _normalise(path: str) -> str:
    text = str(path).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


class PermissionStore:
    """Where grants live. Backed by the Database so 'always' survives a restart."""

    def __init__(self, database: Any) -> None:
        self._db = database

    async def covering(
        self,
        tool_name: str,
        session_id: str,
        arguments: dict | None = None,
    ) -> bool:
        for grant in await self._db.load_permissions(session_id):
            if grant.covers(tool_name, session_id, arguments):
                return True
        return False

    async def save(self, grant: PermissionGrant) -> None:
        await self._db.save_permission(grant)


# ---------------------------------------------------------------------------
# The request and the manager
# ---------------------------------------------------------------------------
@dataclass
class PermissionRequest:
    """Exactly what the user is being asked to approve."""

    call_id: str
    tool: str
    arguments: dict = field(default_factory=dict)
    preview: str = ""
    reason: str = ""


@dataclass
class PermissionAnswer:
    """A decision on one request: yes/no, for how long, and narrowed to where."""

    allowed: bool
    duration: PermissionDuration = PermissionDuration.ONCE
    scope: str = ""


class PermissionResponder(ABC):
    """Where a permission answer comes from.

    This is the one seam the manager needs and the only thing that differs between
    running at a terminal and running behind an API. The manager decides *whether*
    to ask (policy, saved grants) and remembers the answer; a responder only
    supplies the answer for one request. Two of them ship:

      - `ChannelResponder` (here) parks the request and waits for someone else to
        deliver the decision - the API answers a POST, the CLI answers stdin
        through the same `deliver`. It is the default, so nothing has to be wired
        for the async case.
      - `TerminalResponder` (in the CLI) prompts on stdin and answers inline.

    Because the whole surface is `respond`/`deliver`/`pending`, a new channel - a
    chat message, a desktop dialog - is just another implementation; the manager
    and the permission model never change.
    """

    @abstractmethod
    async def respond(self, request: PermissionRequest) -> PermissionAnswer:
        """Return the decision for one request, awaiting it if it isn't here yet."""
        ...

    def deliver(self, call_id: str, answer: PermissionAnswer) -> None:
        """Hand an answer to a waiting `respond`. A no-op unless one is waiting.

        Only channels that wait out-of-band (the async one) need this; a responder
        that answers inline (the terminal) leaves it as the no-op it is here.
        """

    def pending(self) -> list:
        """Requests currently awaiting a decision, for a UI to show. Usually empty."""
        return []


class ChannelResponder(PermissionResponder):
    """Parks each request and waits for the decision to be delivered out-of-band.

    This is the async case the plan calls for: a tool call blocks on `respond`,
    the request shows up in `pending()` for a UI to render, and whoever is driving
    (an API handler, the CLI's stdin loop) calls `deliver` with the answer. It is
    exactly the future-per-call mechanism the manager used to hold itself, lifted
    out so the terminal can take its place without the manager knowing.
    """

    def __init__(self) -> None:
        self._waiting: dict = {}  # call_id -> (future, request)

    async def respond(self, request: PermissionRequest) -> PermissionAnswer:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._waiting[request.call_id] = (future, request)
        try:
            return await future
        finally:
            self._waiting.pop(request.call_id, None)

    def deliver(self, call_id: str, answer: PermissionAnswer) -> None:
        entry = self._waiting.get(call_id)
        if entry is None:
            return  # unknown or already answered - a clean no-op, as the API relies on
        future, _request = entry
        if not future.done():
            future.set_result(answer)

    def pending(self) -> list:
        return [request for (_f, request) in self._waiting.values()]


class PermissionManager:
    """Asks the user when a policy says to, and remembers the answer.

    The manager owns the *decision to ask* and the *memory of the answer* - the
    saved grants, the events on the bus - and delegates only the getting-an-answer
    to a `PermissionResponder`. The default responder waits for the answer to be
    delivered (API or CLI); the CLI can install a terminal one instead. Swapping it
    changes where approvals come from and nothing else.
    """

    def __init__(
        self,
        store: PermissionStore,
        event_bus: Any,
        responder: PermissionResponder | None = None,
    ) -> None:
        self._store = store
        self._bus = event_bus
        self._responder: PermissionResponder = responder or ChannelResponder()

    @property
    def responder(self) -> PermissionResponder:
        return self._responder

    def set_responder(self, responder: PermissionResponder) -> None:
        """Choose where answers come from (e.g. the CLI installs a terminal one)."""
        self._responder = responder

    async def ask(self, request: PermissionRequest, session_id: str, run_id: str = "") -> bool:
        """Return True if the call may proceed. Reuses a saved grant if one covers it."""
        if await self._store.covering(request.tool, session_id, request.arguments):
            await self._bus.emit(
                session_id,
                EventType.PERMISSION_RESOLVED,
                {"call_id": request.call_id, "tool": request.tool, "allowed": True, "reused_grant": True},
                run_id,
            )
            return True

        await self._bus.emit(
            session_id,
            EventType.PERMISSION_REQUESTED,
            {
                "call_id": request.call_id,
                "tool": request.tool,
                "arguments": request.arguments,
                "preview": request.preview,
                "reason": request.reason,
            },
            run_id,
        )

        # The one thing that differs between a terminal and an API: getting the answer.
        answer = await self._responder.respond(request)

        if answer.allowed and answer.duration != PermissionDuration.ONCE:
            grant = PermissionGrant(
                tool_pattern=request.tool,
                duration=answer.duration,
                session_id=session_id if answer.duration == PermissionDuration.SESSION else "",
                argument_pattern=answer.scope.strip(),
            )
            await self._store.save(grant)

        await self._bus.emit(
            session_id,
            EventType.PERMISSION_RESOLVED,
            {
                "call_id": request.call_id,
                "tool": request.tool,
                "allowed": answer.allowed,
                "duration": answer.duration.value,
                "scope": answer.scope.strip(),
            },
            run_id,
        )
        return answer.allowed

    async def resolve(
        self,
        call_id: str,
        allowed: bool,
        duration: PermissionDuration = PermissionDuration.ONCE,
        scope: str = "",
    ) -> None:
        """Deliver a user's answer to a call waiting on the default (channel) responder.

        This is the out-of-band path the API and the CLI's stdin loop use: they see
        a request in `pending()` and answer it here. The grant-saving and the
        `PERMISSION_RESOLVED` event happen back in `ask`, once the answer lands, so
        there is exactly one place that touches the store and the bus. Answering an
        unknown or already-answered call is a clean no-op.

        `scope` narrows a remembered yes to a place: pass "notes" and the grant only
        covers calls whose every path argument is under `notes/`.
        """
        self._responder.deliver(call_id, PermissionAnswer(allowed, duration, scope))

    def pending(self) -> list:
        """The requests currently waiting on the user (for a UI to show)."""
        return self._responder.pending()

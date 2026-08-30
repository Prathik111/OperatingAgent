"""Permissions: policy verdicts, the strictest-wins chain, and grant reuse."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.main import _read_answer
from agent_native.permissions import (
    ChannelResponder,
    PermissionAnswer,
    PermissionDecision,
    PermissionDuration,
    PermissionGrant,
    PermissionManager,
    PermissionRequest,
    PermissionResponder,
    PermissionRule,
    PermissionStore,
    PolicyChain,
    RulePolicy,
    SessionPolicy,
)
from agent_native.tools.base import ToolDefinition, ToolPermissions


def _definition(name: str, **flags) -> ToolDefinition:
    return ToolDefinition(name=name, permissions=ToolPermissions(**flags))


def _ctx(session_id: str) -> SimpleNamespace:
    """A stand-in run context carrying just a session id, for policy tests."""
    return SimpleNamespace(session=SimpleNamespace(id=session_id))


def _manager():
    db = MemoryDatabase()
    return PermissionManager(PermissionStore(db), EventBus(db))


# -- policy verdicts ---------------------------------------------------------
async def test_readonly_is_allowed():
    assert RulePolicy().check(None, _definition("r", read_only=True), {}).result == PermissionDecision.ALLOW


async def test_destructive_is_asked():
    assert RulePolicy().check(None, _definition("w", destructive=True), {}).result == PermissionDecision.ASK


async def test_explicit_deny_rule_wins():
    policy = RulePolicy([PermissionRule(id="no", tool_pattern="danger*", result=PermissionDecision.DENY)])
    assert policy.check(None, _definition("danger_tool", read_only=True), {}).result == PermissionDecision.DENY


async def test_allow_rule_cannot_pierce_destructive_floor():
    # A stray 'allow' rule must not wave a destructive tool through; the floor holds.
    policy = RulePolicy([PermissionRule(id="loose", tool_pattern="write*", result=PermissionDecision.ALLOW)])
    assert policy.check(None, _definition("write_file", destructive=True), {}).result == PermissionDecision.ASK


async def test_rule_can_tighten_a_read_only_floor():
    # read-only floor is ALLOW; an 'ask' rule is stricter, so it takes effect.
    policy = RulePolicy([PermissionRule(id="careful", tool_pattern="r*", result=PermissionDecision.ASK)])
    assert policy.check(None, _definition("r", read_only=True), {}).result == PermissionDecision.ASK


async def test_chain_takes_strictest():
    chain = PolicyChain([RulePolicy(), SessionPolicy()])
    assert chain.check(None, _definition("w", destructive=True), {}).result == PermissionDecision.ASK


async def test_session_policy_asks_first_host_then_allows():
    policy = SessionPolicy()
    first = policy.check(None, _definition("fetch", needs_network=True), {"host": "example.com"})
    second = policy.check(None, _definition("fetch", needs_network=True), {"host": "example.com"})
    assert first.result == PermissionDecision.ASK
    assert second.result == PermissionDecision.ALLOW


async def test_session_policy_memory_is_per_session():
    # Reaching a host in one session must not silently approve it in another.
    policy = SessionPolicy()
    d = _definition("fetch", needs_network=True)
    first_a = policy.check(_ctx("A"), d, {"host": "example.com"})
    second_a = policy.check(_ctx("A"), d, {"host": "example.com"})
    first_b = policy.check(_ctx("B"), d, {"host": "example.com"})
    assert first_a.result == PermissionDecision.ASK
    assert second_a.result == PermissionDecision.ALLOW  # remembered within session A
    assert first_b.result == PermissionDecision.ASK     # session B is asked fresh


async def test_grant_covers_scoping():
    grant = PermissionGrant(tool_pattern="write_file", duration=PermissionDuration.SESSION, session_id="s1")
    assert grant.covers("write_file", "s1")
    assert not grant.covers("write_file", "s2")
    assert not grant.covers("read_file", "s1")


# -- a yes narrowed to a place ------------------------------------------------
#
# "Yes, writes are fine for the rest of the session" is more than most people mean.
# `argument_pattern` is how they say the smaller thing - "writes under notes/" -
# and the rule that keeps it small is that *every* path a call touches has to be
# inside the pattern. Matching on the first one found would turn a narrow yes into
# a wide one, which is the whole failure this is here to avoid.
def _scoped(pattern: str = "notes") -> PermissionGrant:
    return PermissionGrant(
        tool_pattern="filesystem_write_file",
        duration=PermissionDuration.SESSION,
        session_id="s1",
        argument_pattern=pattern,
    )


async def test_a_scoped_grant_covers_inside_and_not_outside():
    grant = _scoped()
    assert grant.covers("filesystem_write_file", "s1", {"path": "notes/today.md"})
    assert not grant.covers("filesystem_write_file", "s1", {"path": "src/main.py"})


async def test_a_folder_pattern_is_not_a_prefix_of_a_siblings_name():
    assert not _scoped().covers("filesystem_write_file", "s1", {"path": "notes-other/x.md"})


async def test_one_uncovered_path_sinks_the_whole_call():
    """The rule is every, not any."""
    grant = _scoped()
    assert not grant.covers("filesystem_write_file", "s1", {"paths": ["notes/a", "/etc/hosts"]})
    assert grant.covers("filesystem_write_file", "s1", {"paths": ["notes/a", "notes/b"]})


async def test_a_call_naming_no_place_is_not_inside_the_place_approved():
    assert not _scoped().covers("filesystem_write_file", "s1", {"content": "no path here"})


async def test_separators_are_normalised():
    assert _scoped().covers("filesystem_write_file", "s1", {"path": "notes\\sub\\deep.md"})
    assert _scoped().covers("filesystem_write_file", "s1", {"path": "./notes/deep.md"})


async def test_a_pattern_with_wildcards_is_used_as_a_glob():
    grant = PermissionGrant("*", PermissionDuration.SESSION, "s1", "*.md")
    assert grant.covers("filesystem_write_file", "s1", {"path": "notes/x.md"})
    assert not grant.covers("filesystem_write_file", "s1", {"path": "notes/x.py"})


async def test_an_unscoped_grant_behaves_as_it_did_before():
    grant = PermissionGrant("filesystem_write_file", PermissionDuration.SESSION, "s1")
    assert grant.covers("filesystem_write_file", "s1", {"path": "anywhere/at/all"})


# -- the manager: ask, resolve, remember -------------------------------------
async def _resolve_when_pending(manager, call_id, allowed, duration=PermissionDuration.ONCE, scope=""):
    while True:
        if manager.pending():
            await manager.resolve(call_id, allowed, duration, scope)
            return
        await asyncio.sleep(0.005)


async def test_ask_resolved_true():
    manager = _manager()
    request = PermissionRequest(call_id="c", tool="write_file", preview="p", reason="r")
    task = asyncio.create_task(_resolve_when_pending(manager, "c", True))
    allowed = await manager.ask(request, "s1")
    await task
    assert allowed is True


async def test_ask_resolved_false():
    manager = _manager()
    request = PermissionRequest(call_id="c", tool="write_file")
    task = asyncio.create_task(_resolve_when_pending(manager, "c", False))
    allowed = await manager.ask(request, "s1")
    await task
    assert allowed is False


async def test_session_grant_is_reused():
    manager = _manager()
    first = PermissionRequest(call_id="c1", tool="write_file")
    task = asyncio.create_task(_resolve_when_pending(manager, "c1", True, PermissionDuration.SESSION))
    allowed_first = await manager.ask(first, "s1")
    await task

    # Second ask for the same tool + session should be auto-allowed, no new prompt.
    second = PermissionRequest(call_id="c2", tool="write_file")
    allowed_second = await manager.ask(second, "s1")

    assert allowed_first and allowed_second
    assert not manager.pending()


async def test_once_grant_is_not_reused():
    manager = _manager()
    first = PermissionRequest(call_id="c1", tool="write_file")
    task = asyncio.create_task(_resolve_when_pending(manager, "c1", True, PermissionDuration.ONCE))
    await manager.ask(first, "s1")
    await task

    # A ONCE grant leaves nothing behind, so the next ask must prompt again.
    second = PermissionRequest(call_id="c2", tool="write_file")
    task2 = asyncio.create_task(_resolve_when_pending(manager, "c2", True, PermissionDuration.ONCE))
    allowed_second = await manager.ask(second, "s1")
    await task2
    assert allowed_second is True


async def test_a_scoped_yes_is_reused_inside_and_asks_again_outside():
    """The whole feature, end to end: one prompt for `notes/`, none for the next
    write under it, and a fresh prompt for a write anywhere else."""
    manager = _manager()

    first = PermissionRequest(
        call_id="c1", tool="filesystem_write_file", arguments={"path": "notes/one.md"}
    )
    task = asyncio.create_task(
        _resolve_when_pending(manager, "c1", True, PermissionDuration.SESSION, "notes")
    )
    assert await manager.ask(first, "s1") is True
    await task

    inside = PermissionRequest(
        call_id="c2", tool="filesystem_write_file", arguments={"path": "notes/two.md"}
    )
    assert await manager.ask(inside, "s1") is True     # returns without anyone answering
    assert not manager.pending()

    outside = PermissionRequest(
        call_id="c3", tool="filesystem_write_file", arguments={"path": "src/main.py"}
    )
    task3 = asyncio.create_task(_resolve_when_pending(manager, "c3", False))
    assert await manager.ask(outside, "s1") is False   # stopped and asked, and refused
    await task3


# -- the responder seam: where the answer comes from -------------------------
#
# The manager decides whether to ask and remembers the answer; a PermissionResponder
# only supplies one. The default (ChannelResponder) parks the request and waits for
# an out-of-band `deliver` - the API answering a POST, the CLI answering stdin. A
# different responder can answer inline instead, and nothing else about the manager
# changes. These pin that seam.
async def test_channel_responder_parks_until_delivered():
    responder = ChannelResponder()
    request = PermissionRequest(call_id="c", tool="write_file")

    async def deliver_soon():
        while not responder.pending():
            await asyncio.sleep(0.005)
        assert responder.pending()[0].call_id == "c"
        responder.deliver("c", PermissionAnswer(True, PermissionDuration.SESSION, "notes"))

    task = asyncio.create_task(deliver_soon())
    answer = await responder.respond(request)
    await task
    assert answer.allowed and answer.duration == PermissionDuration.SESSION and answer.scope == "notes"
    assert not responder.pending()  # cleared once answered


async def test_delivering_to_an_unknown_call_is_a_noop():
    # The API answers a call_id it was handed; a stale or bogus one must not raise.
    ChannelResponder().deliver("nobody", PermissionAnswer(True))


class _InlineResponder(PermissionResponder):
    """A responder that answers on the spot, like the terminal one does."""

    def __init__(self, answer: PermissionAnswer) -> None:
        self._answer = answer
        self.seen: list = []

    async def respond(self, request: PermissionRequest) -> PermissionAnswer:
        self.seen.append(request.call_id)
        return self._answer


async def test_swapping_the_responder_changes_where_answers_come_from():
    """With an inline responder installed, `ask` needs no external `resolve` - and
    the manager still saves the grant and reuses it, because only the source of the
    answer moved, not the model."""
    manager = _manager()
    manager.set_responder(_InlineResponder(PermissionAnswer(True, PermissionDuration.SESSION)))

    first = PermissionRequest(call_id="c1", tool="write_file")
    assert await manager.ask(first, "s1") is True   # answered inline, nobody resolved
    assert not manager.pending()

    # The session grant the inline yes implied is remembered, so the next call to the
    # same tool is auto-allowed without even reaching the responder.
    manager.set_responder(_InlineResponder(PermissionAnswer(False)))  # would deny if asked
    assert await manager.ask(PermissionRequest(call_id="c2", tool="write_file"), "s1") is True


# -- what the terminal accepts as an answer ----------------------------------
async def test_the_terminal_answer_vocabulary():
    """Three answers, not two: once, no, and 'for the session under this folder'.

    Anything unclear is a no. A destructive call is the wrong place to guess at
    what somebody meant.
    """
    once = (True, PermissionDuration.ONCE, "")
    no = (False, PermissionDuration.ONCE, "")
    assert _read_answer("y") == once
    assert _read_answer("yes") == once
    assert _read_answer("") == no
    assert _read_answer("n") == no
    assert _read_answer("maybe later") == no
    assert _read_answer("s") == (True, PermissionDuration.SESSION, "")
    assert _read_answer("s notes") == (True, PermissionDuration.SESSION, "notes")
    assert _read_answer('s "my notes"') == (True, PermissionDuration.SESSION, "my notes")

"""Skills: a named folder of instructions pulled in only when it's relevant.

Step 17's promise is the progressive-disclosure pattern the 2026 harness ships:
`AGENT.md` is always loaded, but a *skill* - a named folder holding a `SKILL.md` -
is listed cheaply in the prompt (its name and a one-line description) and its full
body enters the conversation only when the model invokes it. The economy is the
whole point: the catalogue costs a line per skill, and the instructions are paid
for only when a task actually reaches for one.

The plan's own verify has two halves, and the two e2e tests below are exactly
those halves:

  * ABSENT from the base prompt - the system message names the skill and says what
    it's for, but the body is nowhere in it (`test_base_prompt_lists_the_name_not_the_body`);
  * PRESENT after the model invokes it, and behaviour changes accordingly - a run
    whose model calls `invoke_skill` on turn one has the skill's body on the wire
    by turn two, carried in as the tool result the next request replays
    (`test_invoking_a_skill_brings_its_body_into_the_next_request`).

Everything under those is the machinery that makes the two halves trustworthy:
discovery reads only the name and description (never the body) up front, from
either skill root, deduped and sorted so a prompt built twice reads the same; the
listing is names-not-bodies; the tool reads the body live from disk, recovers from
a bad name instead of failing, and trims a body too large to swamp the window.

Offline by construction: a turn-aware stand-in model, no network, no key. Run
under pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src:packages/agent-native \
        python3 packages/agent-native/tests/test_skills.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent_native.config import (
    AgentConfig,
    PromptBuilder,
    Skill,
    _first_meaningful_line,
    _split_frontmatter,
    discover_skills,
    skill_listing,
)
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    Role,
    Session,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.loop import AgentLoop, Cancellation, Limits, RunContext, RunStatus
from agent_native.permissions import (
    Decision,
    PermissionDecision,
    Policy,
    PolicyChain,
)
from agent_native.service import AgentRuntime, AgentService
from agent_native.tools.base import ToolRegistry
from agent_native.tools.manager import ToolManager
from agent_native.tools.skill_tool import (
    MAX_SKILL_BODY_CHARS,
    InvokeSkillTool,
    _catalogue,
)

from tests._scripted import ScriptedProvider, call_event, scripted_registry, text_event

#: A marker put ONLY in a skill's body, never in its name or description, so a test
#: can tell "the catalogue mentions this skill" apart from "the body was loaded".
SENTINEL = "ZZ_SKILL_BODY_SENTINEL_42"


# ---------------------------------------------------------------------------
# Writing a skills tree on disk
# ---------------------------------------------------------------------------
def _write_skill(base: str, folder: str, text: str, root: str = "skills") -> Path:
    """Create ``<base>/<root>/<folder>/SKILL.md`` with the given text; return its path."""
    manifest = Path(base) / root / folder / "SKILL.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(text, encoding="utf-8")
    return manifest


_GREETER = (
    "---\n"
    "name: greeter\n"
    "description: Greets the user warmly.\n"
    "---\n"
    "# Greeter\n\n"
    f"When asked to greet, follow these steps. {SENTINEL}\n"
)


# ---------------------------------------------------------------------------
# Tiny doubles for the loop e2e: a turn-aware model, allow-all, a must-not-ask prompter
# ---------------------------------------------------------------------------
class _InvokeThenFinish:
    """Turn one invokes a named skill; every turn after finishes with text.

    Records ``(messages, tools)`` per call, exactly like the scripted provider, so
    a test can read back what was on the wire the turn *after* the skill was loaded
    - which is where the body has to appear.
    """

    def __init__(self, skill_name: str) -> None:
        self._skill_name = skill_name
        self.requests: list = []
        self.calls = 0
        self.closed = False

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.requests.append((messages, tools))
        self.calls += 1
        try:
            if self.calls == 1:
                yield call_event(0, "invoke_skill", json.dumps({"name": self._skill_name}))
            else:
                yield text_event("done")
        finally:
            self.closed = True

    def count_tokens(self, messages: list) -> int:
        return 0


class _AllowAll(Policy):
    """A policy that allows everything - loading a skill is read-only, never gated."""

    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        return Decision(PermissionDecision.ALLOW, reason="allow-all")


class _MustNotAsk:
    """A prompter that fails if used: invoking a skill must never prompt the user."""

    def __init__(self) -> None:
        self.asked = 0

    async def ask(self, request: Any, session_id: str) -> bool:
        self.asked += 1
        raise AssertionError("loading a skill is read-only and must never prompt")


class _ToolCtx:
    """The slice of RunContext the tool reads: a session with a working folder."""

    def __init__(self, working_directory: str) -> None:
        self.session = Session(working_directory=working_directory)


def _service(db: MemoryDatabase, provider: ScriptedProvider) -> AgentService:
    """A service on a provider-free runtime save for one scripted model (offline)."""
    runtime = AgentRuntime(
        database=db,
        model_registry=scripted_registry(provider),
        agents=[AgentConfig(name="build", model="scripted-1")],
    )
    return AgentService(runtime)


def _invoke_loop(skill_name: str) -> tuple:
    """A loop wired to the real InvokeSkillTool and a turn-aware provider.

    The same registry is handed to the loop and the ToolManager, so the tool the
    model calls is the tool that runs. Allow-all policy + must-not-ask prompter make
    "the body reached the next turn" the only thing under test, not the gate.
    """
    db = MemoryDatabase()
    provider = _InvokeThenFinish(skill_name)
    registry = scripted_registry(provider)
    tools = ToolRegistry()
    tools.register(InvokeSkillTool())
    prompter = _MustNotAsk()
    manager = ToolManager(tools, PolicyChain([_AllowAll()]), prompter)
    loop = AgentLoop(registry, tools, manager, ContextManager(), EventBus(db), db)
    return loop, db, provider, prompter


def _context(session: Session) -> RunContext:
    return RunContext(
        session=session,
        run_id="run_skill",
        config=AgentConfig(model="scripted-1"),
        limits=Limits(max_turns=5),
        cancellation=Cancellation(),
    )


# ---------------------------------------------------------------------------
# Discovery: only the name and description are read up front
# ---------------------------------------------------------------------------
async def test_discover_reads_name_and_description_from_frontmatter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_skill(tmp, "greeter", _GREETER)
        skills = discover_skills(tmp)

        assert [s.name for s in skills] == ["greeter"]
        skill = skills[0]
        assert skill.description == "Greets the user warmly."
        assert skill.path.endswith("SKILL.md")
        # Discovery reads the manifest but the *body* is not part of the Skill; it
        # is loaded only later, by .body().
        assert SENTINEL not in skill.name
        assert SENTINEL not in skill.description


async def test_discover_falls_back_to_folder_name_and_first_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # No frontmatter at all: the name comes from the folder, the description
        # from the first meaningful line (heading '#' stripped).
        _write_skill(tmp, "beta", "# Beta Helper\n\nDoes beta things.\n")
        # Frontmatter names the skill something other than its folder.
        _write_skill(tmp, "alpha", "---\nname: Custom Name\n---\nbody\n")
        by_name = {s.name: s for s in discover_skills(tmp)}

        assert "Custom Name" in by_name          # frontmatter name wins over folder
        assert "beta" in by_name                 # folder name is the fallback
        assert by_name["beta"].description == "Beta Helper"


async def test_discover_scans_agent_skills_root_dedups_and_sorts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_skill(tmp, "zeta", "---\nname: zeta\ndescription: from skills.\n---\nb\n")
        _write_skill(tmp, "alpha", "---\nname: alpha\ndescription: from agent.\n---\nb\n",
                     root=".agent/skills")
        # A duplicate name present in both roots: 'skills' is scanned first, so it wins.
        _write_skill(tmp, "dup", "---\nname: shared\ndescription: from skills.\n---\nb\n")
        _write_skill(tmp, "dup2", "---\nname: shared\ndescription: from agent.\n---\nb\n",
                     root=".agent/skills")
        skills = discover_skills(tmp)
        names = [s.name for s in skills]

        assert names == sorted(names)                    # sorted by name
        assert "zeta" in names and "alpha" in names       # both roots contribute
        shared = next(s for s in skills if s.name == "shared")
        assert shared.description == "from skills."        # first root wins the dup


async def test_discover_is_empty_and_safe_without_skills() -> None:
    assert discover_skills("") == []                       # no folder at all
    with tempfile.TemporaryDirectory() as tmp:
        assert discover_skills(tmp) == []                  # a folder with no skills root
        # A skills root that exists but holds a folder with no SKILL.md: skipped, no raise.
        (Path(tmp) / "skills" / "notaskill").mkdir(parents=True)
        assert discover_skills(tmp) == []


async def test_skill_body_reads_live_and_strips_frontmatter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _write_skill(tmp, "greeter", _GREETER)
        skill = Skill(name="greeter", description="d", path=str(manifest))

        body = skill.body()
        assert SENTINEL in body                 # the instructions are there...
        assert "description:" not in body       # ...but the frontmatter is stripped
        assert not body.startswith("---")
        # An unreadable path is an empty body, never a raise.
        assert Skill(name="x", description="", path=str(Path(tmp) / "gone.md")).body() == ""


async def test_split_frontmatter_and_first_meaningful_line() -> None:
    meta, body = _split_frontmatter("---\nname: g\ndescription: hi\n---\nBODY\n")
    assert meta == {"name": "g", "description": "hi"} and body.strip() == "BODY"
    # No leading fence: everything is body, no metadata.
    assert _split_frontmatter("no fence here") == ({}, "no fence here")
    # A fence that never closes: treated as bodied, no metadata (not a partial parse).
    assert _split_frontmatter("---\nname: g\nstill open") == ({}, "---\nname: g\nstill open")
    # First meaningful line skips blanks, fences and a leading '#'.
    assert _first_meaningful_line("\n---\n# Title\nrest") == "Title"
    assert _first_meaningful_line("\n\n   \n") == ""


# ---------------------------------------------------------------------------
# The listing: names and one-liners, never bodies
# ---------------------------------------------------------------------------
async def test_skill_listing_names_not_bodies() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_skill(tmp, "greeter", _GREETER)
        listing = skill_listing(discover_skills(tmp))

        assert "greeter" in listing                     # the name is advertised...
        assert "Greets the user warmly." in listing     # ...with its one-liner...
        assert SENTINEL not in listing                  # ...but never the body
        assert "invoke_skill" in listing                # tells the model how to load one

    assert skill_listing([]) == ""                      # nothing to advertise, nothing added


# ---------------------------------------------------------------------------
# The invoke_skill tool: loads a body, recovers from a bad name, trims a huge one
# ---------------------------------------------------------------------------
async def test_invoke_tool_returns_body_for_known_skill() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_skill(tmp, "greeter", _GREETER)
        result = await InvokeSkillTool().execute({"name": "greeter"}, _ToolCtx(tmp))

        assert result.success is True
        assert result.output.startswith("Skill: greeter")
        assert SENTINEL in result.output               # the full instructions come back
        assert result.truncated is False
        # Case-insensitive match: the name is resolved, not used as a path.
        loud = await InvokeSkillTool().execute({"name": "GREETER"}, _ToolCtx(tmp))
        assert SENTINEL in loud.output


async def test_invoke_tool_recovers_from_missing_or_empty_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_skill(tmp, "greeter", _GREETER)
        tool, ctx = InvokeSkillTool(), _ToolCtx(tmp)

        missing = await tool.execute({"name": "nope"}, ctx)
        # A bad name is a recoverable result, not a failure: the model gets the real
        # list back and can pick a name that exists (the body never loads).
        assert missing.success is True
        assert "No skill named 'nope'" in missing.output
        assert "greeter" in missing.output
        assert SENTINEL not in missing.output

        empty = await tool.execute({"name": "   "}, ctx)
        assert empty.success is True and "Which skill?" in empty.output

        # No skills at all: the catalogue says so plainly rather than looking broken.
        assert "No skills are available" in _catalogue([])


async def test_invoke_tool_truncates_a_huge_body() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        big = "---\nname: big\ndescription: huge.\n---\n" + ("x" * (MAX_SKILL_BODY_CHARS + 500))
        _write_skill(tmp, "big", big)
        result = await InvokeSkillTool().execute({"name": "big"}, _ToolCtx(tmp))

        assert result.truncated is True
        assert "[skill truncated]" in result.output
        # The model still gets the opening instructions - a SKILL.md front-loads them.
        assert result.output.startswith("Skill: big")


# ---------------------------------------------------------------------------
# The plan's verify, half one: ABSENT from the base prompt
# ---------------------------------------------------------------------------
async def test_base_prompt_lists_the_name_not_the_body() -> None:
    """create_session seeds the catalogue into the system prompt - names, not bodies."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_skill(tmp, "greeter", _GREETER)
        db = MemoryDatabase()
        service = _service(db, ScriptedProvider([text_event("ok")]))

        session = await service.create_session(agent="build", working_directory=tmp)
        conversation = await db.load_conversation(session.id)
        system = next(m for m in conversation.messages if m.role == Role.SYSTEM).text()

        assert "greeter" in system                      # the skill is advertised...
        assert "Greets the user warmly." in system      # ...with its one-liner...
        assert SENTINEL not in system                   # ...but the body is NOT in the prompt


# ---------------------------------------------------------------------------
# The plan's verify, half two: PRESENT after the model invokes it
# ---------------------------------------------------------------------------
async def test_invoking_a_skill_brings_its_body_into_the_next_request() -> None:
    """A run that invokes a skill has the body on the wire by the next request.

    This is the behaviour change the plan asks for: the body is absent from the
    base prompt the model first sees, and present in the request it sees on the turn
    after it calls invoke_skill - carried in as the tool result the loop replays.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _write_skill(tmp, "greeter", _GREETER)
        loop, db, provider, prompter = _invoke_loop("greeter")
        session = Session(agent="build", working_directory=tmp)
        await db.create_session(session)

        # The real base prompt, catalogue and all, so "absent from it" is a genuine claim.
        prompt = PromptBuilder().build(
            AgentConfig(model="scripted-1"),
            session,
            ["invoke_skill"],
            skills=skill_listing(discover_skills(tmp)),
        )
        conversation = Conversation(
            [system_message(prompt, session.id), user_message(session.id, "greet the user")]
        )
        result = await loop.run(conversation, _context(session))

        assert result.status is RunStatus.FINISHED
        assert prompter.asked == 0                      # loading a skill never prompts
        assert len(provider.requests) == 2              # invoke turn, then finish turn

        turn1 = json.dumps(provider.requests[0][0])     # the base prompt the model first saw
        assert "greeter" in turn1                       # the name was advertised...
        assert SENTINEL not in turn1                    # ...but the body was absent

        turn2 = json.dumps(provider.requests[1][0])     # the request after the invoke
        assert SENTINEL in turn2                         # the body is now on the wire


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_discover_reads_name_and_description_from_frontmatter,
        test_discover_falls_back_to_folder_name_and_first_line,
        test_discover_scans_agent_skills_root_dedups_and_sorts,
        test_discover_is_empty_and_safe_without_skills,
        test_skill_body_reads_live_and_strips_frontmatter,
        test_split_frontmatter_and_first_meaningful_line,
        test_skill_listing_names_not_bodies,
        test_invoke_tool_returns_body_for_known_skill,
        test_invoke_tool_recovers_from_missing_or_empty_name,
        test_invoke_tool_truncates_a_huge_body,
        test_base_prompt_lists_the_name_not_the_body,
        test_invoking_a_skill_brings_its_body_into_the_next_request,
    ]
    failures: list = []
    for test in tests:
        try:
            asyncio.run(test())
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - skills:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - skills: {len(tests)} tests (discovery x5, split/first-line, "
          "listing, tool x3, e2e absent-from-prompt, e2e present-after-invoke).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

"""The loop, end to end, against the real Groq model.

These are **live** tests: they call Groq over the network and cost tokens. Each one
skips itself when `groq` isn't installed or `GROQ_API_KEY` isn't set, so the suite
stays runnable on a machine that can't reach Groq.

Because a real model chooses its own path, these assert on invariants that must
hold no matter what it decides - the run ends cleanly, the conversation stays
legal, a failed tool comes back as something it can read - rather than on an exact
script of turns. The fine-grained, adversarial cases (a tool that doesn't exist,
malformed arguments, a denied call) are covered offline in test_tools.py, which
exercises the same code path through ToolManager without needing a model.
"""

from __future__ import annotations

import os
import tempfile

from agent_native.events import EventType
from agent_native.loop import Limits, RunStatus
from agent_native.models.base import StreamType
from tests._helpers import make_runtime, require_live_groq, run_with_auto_permissions


class RecordingProvider:
    """Wraps the real provider and records the wire messages it was handed.

    Not a stand-in for the model - every call still goes to Groq. This only lets a
    test inspect exactly what was put on the wire each turn.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.seen_messages: list = []
        self.tool_format = getattr(inner, "tool_format", None)

    async def stream(self, messages, tools, model, temperature=0.0):
        self.seen_messages.append(messages)
        async for event in self._inner.stream(messages, tools, model, temperature):
            yield event

    def count_tokens(self, messages: list) -> int:
        return self._inner.count_tokens(messages)


def _workdir_with(name: str, content: str) -> str:
    workdir = tempfile.mkdtemp()
    with open(os.path.join(workdir, name), "w") as fh:
        fh.write(content)
    return workdir


def _tool_names_used(conversation) -> list:
    names = []
    for msg in conversation.messages:
        for call in msg.tool_calls():
            names.append(call.name)
    return names


async def test_end_to_end_reads_a_file_and_answers():
    """A real task start to finish: it reads the file, then answers from it."""
    require_live_groq()
    workdir = _workdir_with("config.txt", "port=8080\nhost=localhost\n")
    runtime, service = make_runtime(workdir, max_turns=6)
    session = await service.create_session(agent="build", working_directory=workdir)

    result = await run_with_auto_permissions(
        service,
        session.id,
        "Read the file config.txt and tell me which port it uses.",
        limits=Limits(max_turns=6),
    )

    assert result.status == RunStatus.FINISHED, result.error
    assert "8080" in result.final_text
    assert result.usage.input_tokens > 0  # a real call reported real usage

    conv = await service.get_conversation(session.id)
    assert conv.is_valid()                          # every tool call kept its result
    assert "read_file" in _tool_names_used(conv)

    events = await runtime.database.load_events(session.id, 0)
    seqs = [e.sequence for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))               # strictly monotonic, no repeats
    assert EventType.RUN_FINISHED in {e.type for e in events}
    assert len(runtime.database._runs) == 1


async def test_a_failing_tool_is_an_observation_not_a_crash():
    """Reading a missing file fails; the model reads the failure and still answers."""
    require_live_groq()
    workdir = tempfile.mkdtemp()  # deliberately empty
    runtime, service = make_runtime(workdir, max_turns=6)
    session = await service.create_session(agent="build", working_directory=workdir)

    result = await run_with_auto_permissions(
        service,
        session.id,
        "Read the file does_not_exist.txt and tell me what it says.",
        limits=Limits(max_turns=6),
    )

    # The tool failed, but the run finished cleanly rather than throwing.
    assert result.status == RunStatus.FINISHED, result.error
    conv = await service.get_conversation(session.id)
    assert conv.is_valid()

    tool_messages = [m for m in conv.messages if m.role.value == "tool"]
    assert tool_messages, "expected the model to try reading the file"
    failed = [c for m in tool_messages for c in m.tool_calls() if c.error]
    assert failed, "expected a failed tool result to be recorded"


async def test_the_turn_limit_stops_the_run():
    """With one turn allowed and a task needing a tool, the run stops at the limit."""
    require_live_groq()
    workdir = _workdir_with("config.txt", "port=8080\n")
    runtime, service = make_runtime(workdir, max_turns=1)
    session = await service.create_session(agent="build", working_directory=workdir)

    result = await run_with_auto_permissions(
        service,
        session.id,
        "Read config.txt with the read_file tool, then summarize it.",
        limits=Limits(max_turns=1),
    )

    assert result.status == RunStatus.LIMIT_REACHED
    assert result.turns == 1


async def test_the_model_sees_a_legal_conversation_each_turn():
    """Whatever the model does, what goes on the wire must stay well-formed."""
    require_live_groq()
    workdir = _workdir_with("config.txt", "port=8080\n")
    runtime, service = make_runtime(workdir, max_turns=6)

    # Wrap the real provider so we can inspect the wire without replacing the model.
    spy = RecordingProvider(runtime.models.get_provider(runtime.models.get("llama-3.3-70b")))
    runtime.models.register_provider("groq", spy)

    session = await service.create_session(agent="build", working_directory=workdir)
    await run_with_auto_permissions(
        service,
        session.id,
        "Read config.txt and tell me the port.",
        limits=Limits(max_turns=6),
    )

    assert spy.seen_messages, "the provider was never called"
    for wire in spy.seen_messages:
        assert wire[0]["role"] == "system"
        # Every tool message must answer a tool call that was announced earlier.
        announced = {
            call["id"]
            for msg in wire
            if msg["role"] == "assistant"
            for call in (msg.get("tool_calls") or [])
        }
        for msg in wire:
            if msg["role"] == "tool":
                assert msg["tool_call_id"] in announced


async def test_monitoring_records_run_turn_and_tool_spans():
    """The loop opens a span per run, per turn and per tool - the wiring is live."""
    require_live_groq()
    workdir = _workdir_with("config.txt", "port=8080\n")
    runtime, service = make_runtime(workdir, max_turns=6)
    runtime.monitoring.enabled = True  # the same instance the loop holds
    session = await service.create_session(agent="build", working_directory=workdir)

    await run_with_auto_permissions(
        service,
        session.id,
        "Read config.txt with the read_file tool and tell me the port.",
        limits=Limits(max_turns=6),
    )

    names = [s.name for s in runtime.monitoring.spans]
    assert "run" in names
    assert "turn" in names
    assert "tool" in names


async def test_unknown_model_becomes_a_clean_error():
    """A misconfigured model name ends as an ERROR result, not an uncaught exception.

    This never reaches the network - it fails while resolving the model - so it runs
    everywhere, with or without a key.
    """
    workdir = tempfile.mkdtemp()
    runtime, service = make_runtime(workdir, max_turns=4)
    session = await service.create_session(agent="build", working_directory=workdir)

    runtime.config_for("build").model = "ghost-model"  # nobody registered this

    result = await service.send_message(session.id, "go", limits=Limits(max_turns=4))

    assert result.status == RunStatus.ERROR
    assert "ghost-model" in result.error
    # The failure was still recorded and announced, not swallowed.
    assert len(runtime.database._runs) == 1
    events = await runtime.database.load_events(session.id, 0)
    assert any(e.type == EventType.ERROR for e in events)
    assert any(e.type == EventType.RUN_FINISHED for e in events)


async def test_stream_events_are_the_shape_the_loop_expects():
    """A guard on the provider contract: StreamType members the loop switches on."""
    for member in ("TEXT", "REASONING", "TOOL_CALL", "USAGE", "DONE"):
        assert hasattr(StreamType, member)

"""Context compaction: it must save space without ever splitting a tool pair."""

from __future__ import annotations

from agent_native.context import ContextManager, write_summary
from agent_native.conversation import (
    Conversation,
    Role,
    ToolCall,
    ToolCallStatus,
    assistant_message,
    system_message,
    tool_result_message,
    user_message,
)
from agent_native.models.base import Model
from tests._scripted import ScriptedProvider, text_event


def _tiny(context_size: int = 10) -> Model:
    """A model description with a deliberately tiny window. No API calls here."""
    return Model(provider="groq", model_id="llama-3.3-70b", context_size=context_size)


def _long_conversation() -> Conversation:
    conv = Conversation()
    conv.add(system_message("sys"))
    conv.add(user_message("s", "Please read config.txt and tell me the port."))
    call = ToolCall(
        id="c1", name="read_file", arguments={"path": "config.txt"},
        status=ToolCallStatus.SUCCESS, output="port=8080",
    )
    conv.add(assistant_message("s", text="reading", tool_calls=[call]))
    conv.add(tool_result_message("s", call))
    for i in range(8):
        conv.add(user_message("s", f"q{i}"))
        conv.add(assistant_message("s", text=f"a{i}"))
    return conv


async def test_needs_compaction_threshold():
    conv = _long_conversation()
    manager = ContextManager(threshold=0.8)
    assert manager.needs_compaction(conv, _tiny(10)) is True
    assert manager.needs_compaction(conv, _tiny(100_000)) is False


async def test_needs_compaction_prefers_observed_tokens():
    """A real prompt-token count from the provider beats the local estimate."""
    conv = _long_conversation()
    manager = ContextManager(threshold=0.8)

    # The estimate alone would say "compact" on a tiny window; a real count that
    # is comfortably under the threshold overrides it and says "not yet".
    assert manager.needs_compaction(conv, _tiny(10)) is True
    assert manager.needs_compaction(conv, _tiny(10), observed_input_tokens=1) is False

    # And the reverse: a big real count trips the threshold even though the
    # estimate on a huge window would not.
    assert manager.needs_compaction(conv, _tiny(100_000)) is False
    assert manager.needs_compaction(conv, _tiny(100_000), observed_input_tokens=90_000) is True

    # Zero / None mean "no report yet" and fall back to the estimate.
    assert manager.needs_compaction(conv, _tiny(10), observed_input_tokens=0) is True
    assert manager.needs_compaction(conv, _tiny(10), observed_input_tokens=None) is True


async def test_compact_keeps_valid_and_shape():
    conv = _long_conversation()
    before = len(conv.messages)
    manager = ContextManager(recent_window=6, threshold=0.8)
    result = manager.compact(conv, _tiny(10))

    assert result is not None
    assert len(conv.messages) < before
    assert conv.is_valid()
    assert conv.messages[0].role == Role.SYSTEM       # the original system prompt
    assert conv.messages[1].role == Role.SYSTEM        # the summary that replaced older turns
    assert "summarized" in conv.messages[1].text()
    assert "read_file" in conv.messages[1].text()      # the summary names the tools used


async def test_compact_returns_none_when_nothing_to_fold():
    conv = Conversation()
    conv.add(system_message("sys"))
    conv.add(user_message("s", "hi"))
    manager = ContextManager(recent_window=6)
    assert manager.compact(conv, _tiny(10)) is None


async def test_split_never_breaks_a_tool_pair():
    conv = Conversation()
    conv.add(system_message("sys"))
    for i in range(5):
        conv.add(user_message("s", f"q{i}"))
        conv.add(assistant_message("s", text=f"a{i}"))
    # A tool pair placed right at the recent-window boundary.
    call = ToolCall(
        id="cx", name="write_file", arguments={"path": "x"},
        status=ToolCallStatus.SUCCESS, output="ok",
    )
    conv.add(assistant_message("s", text="writing", tool_calls=[call]))
    conv.add(tool_result_message("s", call))
    conv.add(user_message("s", "thanks"))
    conv.add(assistant_message("s", text="welcome"))

    manager = ContextManager(recent_window=4)
    split = manager.protect_recent_messages(conv)
    older = conv.messages[1:split]
    recent = conv.messages[split:]

    assert Conversation(older).is_valid()
    assert Conversation(recent).is_valid()
    assert manager.check_tool_pairs(older, recent)
    # The write_file request and its result end up on the same side.
    recent_roles = [m.role.value for m in recent]
    assert recent_roles.count("tool") == 1


# ---------------------------------------------------------------------------
# The better summary: the model writes it, the template catches the fall
#
# `compact()` above is a template - honest, deterministic, and a *list* of what
# happened. `compact_with_model()` asks the model to write prose instead, which is
# what a later turn can actually work from. The thing worth testing hardest isn't
# the good path: it's that every way the model call can go wrong still ends in a
# compacted conversation, because compaction runs exactly when the next request
# would otherwise be too long to send.
# ---------------------------------------------------------------------------
def _failed_build_messages() -> list:
    call = ToolCall(
        id="c1", name="terminal_run_command", arguments={"command": "make"},
        status=ToolCallStatus.ERROR, error="missing header foo.h",
    )
    return [
        user_message("s", "Find why the build fails."),
        assistant_message("s", text="checking", tool_calls=[call]),
        tool_result_message("s", call),
    ]


def _conversation_to_fold() -> Conversation:
    conv = Conversation([system_message("sys")])
    for msg in _failed_build_messages():
        conv.add(msg)
    for i in range(8):
        conv.add(user_message("s", f"q{i}"))
    return conv


async def test_the_model_is_sent_the_transcript_as_text_with_no_tools():
    """Tools are deliberately not declared: replaying an assistant's tool calls at
    a provider that wasn't told about those tools is rejected by some of them."""
    provider = ScriptedProvider([text_event("The build fails because foo.h is missing.")])
    summary = await write_summary(_failed_build_messages(), _tiny(100_000), provider)

    assert "foo.h" in summary
    messages, tools = provider.requests[0]
    assert tools == []
    assert messages[0]["role"] == "system" and "carry on without it" in messages[0]["content"]
    # The failing command and its error survive into what the model is asked about.
    assert "terminal_run_command" in messages[1]["content"]
    assert "missing header foo.h" in messages[1]["content"]


async def test_a_provider_that_throws_or_says_nothing_gives_back_nothing():
    older = _failed_build_messages()
    assert await write_summary(older, _tiny(100_000), ScriptedProvider([], fail=True)) == ""
    assert await write_summary(older, _tiny(100_000), ScriptedProvider([])) == ""


async def test_compact_with_model_uses_the_model_and_keeps_the_prefix_first():
    conv = _conversation_to_fold()
    provider = ScriptedProvider([text_event("Build broken: foo.h missing. Next: the include path.")])

    result = await ContextManager(recent_window=4).compact_with_model(conv, _tiny(10), provider)

    assert result is not None
    assert conv.messages[0].role == Role.SYSTEM        # the original prompt, untouched
    assert conv.messages[1].role == Role.SYSTEM        # the summary, after it
    assert "foo.h" in conv.messages[1].text()
    assert conv.is_valid()


async def test_a_failed_model_call_still_compacts():
    """The whole reason this is safe to turn on: the fallback is the old path."""
    conv = _conversation_to_fold()
    result = await ContextManager(recent_window=4).compact_with_model(
        conv, _tiny(10), ScriptedProvider([], fail=True)
    )

    assert result is not None
    assert "summarized" in conv.messages[1].text()     # the template wrote it
    assert conv.is_valid()


async def test_no_provider_at_all_still_compacts():
    conv = _conversation_to_fold()
    result = await ContextManager(recent_window=4).compact_with_model(conv, _tiny(10), None)

    assert result is not None
    assert "summarized" in conv.messages[1].text()

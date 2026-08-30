"""The domain model and the one wire-format author: Conversation.render()."""

from __future__ import annotations

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
from agent_native.models.base import ToolFormat


def _paired_conversation() -> Conversation:
    conv = Conversation()
    conv.add(system_message("sys"))
    conv.add(user_message("s", "hi"))
    request = ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    conv.add(assistant_message("s", text="reading", tool_calls=[request]))
    done = ToolCall(
        id="c1", name="read_file", arguments={"path": "a.txt"},
        status=ToolCallStatus.SUCCESS, output="hello",
    )
    conv.add(tool_result_message("s", done))
    return conv


async def test_render_native_shape():
    wire = _paired_conversation().render()
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[1] == {"role": "user", "content": "hi"}
    assistant = wire[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "reading"
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "hello"}


async def test_is_valid_true_when_paired():
    assert _paired_conversation().is_valid() is True


async def test_is_valid_false_when_unpaired():
    conv = Conversation()
    conv.add(assistant_message("s", text="x", tool_calls=[ToolCall(id="z", name="t")]))
    assert conv.is_valid() is False


async def test_tool_error_renders_error_content():
    conv = Conversation()
    conv.add(tool_result_message("s", ToolCall(id="c1", name="t", status=ToolCallStatus.ERROR, error="boom")))
    assert conv.render()[0]["content"] == "boom"


async def test_assistant_without_text_has_null_content():
    conv = Conversation()
    conv.add(assistant_message("s", tool_calls=[ToolCall(id="c1", name="t")]))
    conv.add(tool_result_message("s", ToolCall(id="c1", name="t", status=ToolCallStatus.SUCCESS, output="ok")))
    assert conv.render()[0]["content"] is None


async def test_render_rejects_non_native():
    raised = False
    try:
        _paired_conversation().render(ToolFormat.TEXT)
    except NotImplementedError:
        raised = True
    assert raised


async def test_factories_set_roles():
    assert user_message("s", "x").role == Role.USER
    assert system_message("x").role == Role.SYSTEM
    assert assistant_message("s", text="x").role == Role.ASSISTANT
    assert tool_result_message("s", ToolCall(id="c", name="t")).role == Role.TOOL

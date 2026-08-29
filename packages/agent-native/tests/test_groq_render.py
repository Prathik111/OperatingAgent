"""The Groq provider's chunk -> StreamEvent conversion, tested with a fake client.

No `groq` library and no network: we inject a stand-in client whose streamed
chunks have the same attribute shape Groq's SDK produces, and check the provider
turns them into the same StreamEvents the loop expects. The second test proves
those streamed tool-call fragments reassemble into a real tool call.
"""

from __future__ import annotations

from agent_native.loop import _build_tool_calls, _merge_fragment
from agent_native.models.base import StreamType
from agent_native.models.groq_model import GROQ_MODELS, Groq


# -- a minimal stand-in for the pieces of Groq's streaming SDK we read --------
class _Fn:
    def __init__(self, name=None, arguments=""):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, index, id, name, arguments):
        self.index = index
        self.id = id
        self.function = _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, reasoning=None, tool_calls=None):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Chunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Completions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _Stream(self._chunks)


class _Chat:
    def __init__(self, chunks):
        self.completions = _Completions(chunks)


class _Client:
    def __init__(self, chunks):
        self.chat = _Chat(chunks)


async def test_groq_stream_converts_chunks():
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="Hello"))]),
        _Chunk(choices=[_Choice(_Delta(reasoning="thinking"))]),
        _Chunk(choices=[_Choice(_Delta(tool_calls=[_Call(0, "id1", "read_file", '{"path":')]))]),
        _Chunk(choices=[_Choice(_Delta(tool_calls=[_Call(0, None, None, '"a.txt"}')]), finish_reason="tool_calls")]),
        _Chunk(usage=_Usage(10, 5)),
    ]
    groq = Groq(api_key="test")
    groq._client = _Client(chunks)  # inject, so _get_client never touches the real lib

    events = []
    async for event in groq.stream([{"role": "user", "content": "hi"}], [], GROQ_MODELS["llama-3.3-70b"]):
        events.append(event)

    types = [e.type for e in events]
    assert StreamType.TEXT in types
    assert StreamType.REASONING in types
    assert StreamType.TOOL_CALL in types
    assert StreamType.USAGE in types
    assert events[-1].type == StreamType.DONE
    assert events[-1].data["finish_reason"] == "tool_calls"

    usage = [e for e in events if e.type == StreamType.USAGE][0]
    assert usage.data["input_tokens"] == 10
    assert usage.data["output_tokens"] == 5

    tool_events = [e for e in events if e.type == StreamType.TOOL_CALL]
    assert tool_events[0].data["index"] == 0
    assert tool_events[0].data["name"] == "read_file"


async def test_groq_fragments_reassemble_into_a_tool_call():
    """The two streamed fragments above must merge back into one real tool call."""
    fragments: dict = {}
    order: list = []
    _merge_fragment(fragments, order, {"index": 0, "id": "id1", "name": "read_file", "arguments": '{"path":'})
    _merge_fragment(fragments, order, {"index": 0, "id": None, "name": None, "arguments": '"a.txt"}'})

    calls = _build_tool_calls(fragments, order)
    assert len(calls) == 1
    assert calls[0].id == "id1"
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.txt"}

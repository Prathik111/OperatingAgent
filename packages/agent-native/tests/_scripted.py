"""A model that says exactly what a test tells it to say.

The tests in test_loop.py call the real Groq model, which is the right way to
check that the loop works against something that thinks - and the wrong way to
check what happens when the user stops a run between the second and third chunk
of a reply. That needs a provider whose timing the test owns.

So this is a stand-in with the same two methods as the real one (`stream` and
`count_tokens`), no network and no key. `on_chunk` is called just before each
event is handed over, which is the hook a test uses to flip a cancellation
exactly where it wants it. `fail=True` makes the call raise instead, for the
paths that have to survive a provider having a bad moment.

Not a test module itself - the name doesn't start with test_.
"""

from __future__ import annotations

from typing import Any

from agent_native.models.base import Model, ModelRegistry, StreamEvent, StreamType


def text_event(text: str) -> StreamEvent:
    return StreamEvent(StreamType.TEXT, {"text": text})


def call_event(index: int, name: str, arguments: str) -> StreamEvent:
    """One complete tool-call fragment. The loop stitches these by index."""
    return StreamEvent(
        StreamType.TOOL_CALL,
        {"index": index, "id": f"call_{index}", "name": name, "arguments": arguments},
    )


class ScriptedProvider:
    """Yields the events it was given, one at a time, and remembers the request."""

    def __init__(self, events: list, on_chunk: Any = None, fail: bool = False) -> None:
        self._events = list(events)
        self.on_chunk = on_chunk
        self._fail = fail
        #: (messages, tools) for every call made, so a test can inspect the wire.
        self.requests: list = []
        #: True once the stream has been closed - the loop must not leave one open.
        self.closed = False

    async def stream(self, messages: list, tools: list, model: Model, temperature: float = 0.0):
        self.requests.append((messages, tools))
        if self._fail:
            raise RuntimeError("the provider is having a bad moment")
        try:
            for event in self._events:
                if self.on_chunk is not None:
                    self.on_chunk()
                yield event
        finally:
            self.closed = True

    def count_tokens(self, messages: list) -> int:
        return 0


def scripted_model(context_size: int = 100_000) -> Model:
    return Model(provider="scripted", model_id="scripted-1", context_size=context_size)


def scripted_registry(provider: ScriptedProvider, name: str = "scripted-1") -> ModelRegistry:
    """A registry wired to one scripted provider, under the model name given."""
    registry = ModelRegistry()
    registry.register_provider("scripted", provider)
    registry.register_model(name, scripted_model())
    return registry

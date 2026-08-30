"""Multimodal input: images (and documents) entering a Conversation, reaching a
vision-capable model, and being refused cleanly by a text-only one.

Step 20's promise, and the plan's own verify, has two halves:

  * **Send an image to a vision-capable model and get a grounded answer.** A user
    turn can carry a `Media` part; `render` turns it into the OpenAI/Groq
    multimodal shape (a `content` list with an `image_url`), the Ollama adapter
    splits that into its own `images` field, and a model marked `supports_vision`
    reads it. `test_vision_model_gets_a_grounded_answer` runs one end-to-end and
    checks the answer is derived from the image that reached the model.
  * **A text-only model reports the input unsupported rather than crashing.**
    `require_vision_support` is the one gate; each real adapter calls it at the top
    of `stream`, before it builds a client, so the refusal is a clear error, not an
    opaque crash deep in a provider call. Verified two ways: the real Groq and
    Ollama adapters raise on the first step (offline, no client), and a run whose
    model can't see images ends in a clean ERROR
    (`test_text_only_model_reports_unsupported_rather_than_crashing`).

Everything else pins the pieces: `media_part` encoding, the render shapes (plain
string when text-only - the shape every other test asserts - a typed list only
when media is present, a document degrading to a labelled note), the Ollama
translation, wire-media detection, and the per-image budgeting estimate.

Offline by construction: scripted stand-in models, no network, no key. Run under
pytest, or straight on a box without it:
    PYTHONPATH=packages/agent-native/src:packages/agent-native \
        python3 packages/agent-native/tests/test_multimodal.py
"""

from __future__ import annotations

import asyncio
import base64
import sys
from typing import Any

from agent_native.config import AgentConfig
from agent_native.context import ContextManager
from agent_native.conversation import (
    Conversation,
    PartType,
    Role,
    Session,
    _MEDIA_TOKEN_ESTIMATE,
    media_part,
    system_message,
    user_message,
)
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.loop import AgentLoop, Cancellation, Limits, RunContext, RunStatus
from agent_native.models.base import (
    Model,
    ModelRegistry,
    StreamEvent,
    StreamType,
    require_vision_support,
    wire_has_media,
)
from agent_native.models.ollama_model import _data_url_to_base64, _to_ollama_messages
from agent_native.permissions import Decision, PermissionDecision, Policy, PolicyChain
from agent_native.tools.base import ToolRegistry
from agent_native.tools.manager import ToolManager

from tests._scripted import text_event

_PNG = b"\x89PNG\r\n\x1a\n small fake image bytes"
_PNG_B64 = base64.b64encode(_PNG).decode("ascii")


def _usage_event(input_tokens: int, output_tokens: int) -> StreamEvent:
    return StreamEvent(
        StreamType.USAGE, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


def _image_wire(text: str = "look") -> list:
    """A rendered user turn carrying one image, the OpenAI/Groq shape."""
    content: list = []
    if text:
        content.append({"type": "text", "text": text})
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}}
    )
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Stand-in providers whose behaviour the test owns
# ---------------------------------------------------------------------------
class _VisionEcho:
    """A vision-capable stand-in: it reads the image off the wire and answers with
    something derived from it, which is how the test knows the media *reached* it
    rather than being dropped somewhere on the way."""

    def __init__(self) -> None:
        self.calls = 0
        self.saw_image = False
        self.seen_mime = ""

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    self.saw_image = True
                    if url.startswith("data:") and ";" in url:
                        self.seen_mime = url[len("data:") : url.index(";")]
        yield _usage_event(10, 10)
        answer = (
            f"I can see the {self.seen_mime} image you sent."
            if self.saw_image
            else "I did not receive an image."
        )
        yield text_event(answer)

    def count_tokens(self, messages: list) -> int:
        return 0


class _GatingTextModel:
    """A provider that gates on vision support exactly like the real adapters do,
    then would answer. Given a text-only model and an image, the gate fires - which
    is the behaviour a run has to turn into a clean ERROR rather than a crash."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, messages: list, tools: list, model: Any, temperature: float = 0.0, **kwargs: Any
    ):
        self.calls += 1
        require_vision_support(messages, model)  # the real-adapter gate, mirrored
        yield _usage_event(5, 5)
        yield text_event("a text answer")

    def count_tokens(self, messages: list) -> int:
        return 0


# ---------------------------------------------------------------------------
# Tiny doubles for the loop wiring (same shape as the routing tests use)
# ---------------------------------------------------------------------------
class _AllowAll(Policy):
    def check(self, context: Any, definition: Any, arguments: dict) -> Decision:
        return Decision(PermissionDecision.ALLOW, reason="allow-all")


class _MustNotAsk:
    async def ask(self, request: Any, session_id: str) -> bool:
        raise AssertionError("these runs have no tools and must never prompt")


def _model(name: str, provider: str, vision: bool = False) -> Model:
    return Model(
        provider=provider, model_id=name, context_size=100_000, supports_vision=vision
    )


def _loop(reg: ModelRegistry):
    db = MemoryDatabase()
    tools = ToolRegistry()
    manager = ToolManager(tools, PolicyChain([_AllowAll()]), _MustNotAsk())
    loop = AgentLoop(reg, tools, manager, ContextManager(), EventBus(db), db)
    return loop, db


def _context(session: Session, config: AgentConfig) -> RunContext:
    return RunContext(
        session=session,
        run_id="run_media",
        config=config,
        limits=Limits(max_turns=3, max_retries=0),
        cancellation=Cancellation(),
    )


# ---------------------------------------------------------------------------
# media_part: how bytes become a carryable, serializable part
# ---------------------------------------------------------------------------
async def test_media_part_encodes_bytes() -> None:
    part = media_part(_PNG)
    assert part.is_image is True
    assert part.mime_type == "image/png"                 # the default
    assert part.data == _PNG_B64                          # bytes were base64-encoded
    assert part.data_url() == f"data:image/png;base64,{_PNG_B64}"


async def test_media_part_keeps_a_base64_string_as_is() -> None:
    part = media_part("QUJD", mime_type="image/jpeg")
    assert part.data == "QUJD"                            # a str is taken as already-encoded
    assert part.mime_type == "image/jpeg"


async def test_document_media_is_not_an_image() -> None:
    part = media_part(b"%PDF-1.7", mime_type="application/pdf")
    assert part.is_image is False


# ---------------------------------------------------------------------------
# The message factory: text, media, or both
# ---------------------------------------------------------------------------
async def test_user_message_carries_text_and_media() -> None:
    msg = user_message("s", "look", media=[media_part(_PNG)])
    assert msg.role == Role.USER
    assert msg.text() == "look"
    assert [p.part_type for p in msg.parts] == [PartType.TEXT, PartType.MEDIA]


async def test_media_only_user_message_has_no_empty_text_part() -> None:
    msg = user_message("s", media=[media_part(_PNG)])
    assert msg.text() == ""
    assert [p.part_type for p in msg.parts] == [PartType.MEDIA]


# ---------------------------------------------------------------------------
# render: a plain string when text-only, a typed list only when media is present
# ---------------------------------------------------------------------------
async def test_render_text_only_user_is_still_a_plain_string() -> None:
    """The backward-compatibility guard: no media -> content is the string it
    always was, which test_conversation.test_render_native_shape also asserts."""
    conv = Conversation([user_message("s", "hi")])
    assert conv.render()[0] == {"role": "user", "content": "hi"}


async def test_render_user_with_image_is_the_openai_content_list() -> None:
    conv = Conversation([user_message("s", "look", media=[media_part(_PNG)])])
    content = conv.render()[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
    }


async def test_render_image_only_omits_the_text_entry() -> None:
    conv = Conversation([user_message("s", media=[media_part(_PNG)])])
    content = conv.render()[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "image_url"


async def test_render_passes_the_detail_hint_through() -> None:
    conv = Conversation(
        [user_message("s", "look", media=[media_part(_PNG, detail="low")])]
    )
    image = conv.render()[0]["content"][1]
    assert image["image_url"]["detail"] == "low"


async def test_render_document_degrades_to_a_text_note() -> None:
    """A non-image has no portable content type across these providers, so it
    becomes a labelled text note rather than an image_url a provider would reject."""
    conv = Conversation(
        [user_message("s", "read this", media=[media_part(b"%PDF", "application/pdf")])]
    )
    content = conv.render()[0]["content"]
    assert content[0] == {"type": "text", "text": "read this"}
    assert content[1]["type"] == "text"
    assert "application/pdf" in content[1]["text"]
    assert not any(p.get("type") == "image_url" for p in content)


async def test_estimate_tokens_counts_an_image() -> None:
    session = Session()
    text_only = Conversation([user_message(session.id, "hello there friend")])
    with_image = Conversation(
        [user_message(session.id, "hello there friend", media=[media_part(_PNG)])]
    )
    delta = with_image.estimate_tokens() - text_only.estimate_tokens()
    assert delta == _MEDIA_TOKEN_ESTIMATE


# ---------------------------------------------------------------------------
# The Ollama translation: OpenAI content-list -> plain text + an `images` field
# ---------------------------------------------------------------------------
async def test_ollama_splits_an_image_into_the_images_field() -> None:
    wire = [{"role": "system", "content": "sys"}] + _image_wire("look")
    out = _to_ollama_messages(wire)
    user = out[1]
    assert user["role"] == "user"
    assert user["content"] == "look"                 # text pulled back to a plain string
    assert user["images"] == [_PNG_B64]              # image lifted into `images`, prefix stripped


async def test_ollama_leaves_a_plain_text_turn_untouched() -> None:
    out = _to_ollama_messages([{"role": "user", "content": "just text"}])
    assert out[0] == {"role": "user", "content": "just text"}
    assert "images" not in out[0]


async def test_data_url_prefix_is_stripped() -> None:
    assert _data_url_to_base64(f"data:image/png;base64,{_PNG_B64}") == _PNG_B64
    assert _data_url_to_base64(_PNG_B64) == _PNG_B64      # already bare, unchanged
    assert _data_url_to_base64("") == ""


# ---------------------------------------------------------------------------
# Detecting media on the wire, and the one gate
# ---------------------------------------------------------------------------
async def test_wire_has_media_detects_an_image_url() -> None:
    assert wire_has_media(_image_wire()) is True


async def test_wire_has_media_is_false_for_text_only() -> None:
    assert wire_has_media([{"role": "user", "content": "hi"}]) is False


async def test_wire_has_media_detects_the_ollama_images_field() -> None:
    assert wire_has_media([{"role": "user", "content": "hi", "images": [_PNG_B64]}]) is True


async def test_require_vision_support_raises_for_a_text_only_model() -> None:
    message = ""
    try:
        require_vision_support(_image_wire(), _model("t", "p"))  # vision defaults False
    except RuntimeError as exc:
        message = str(exc)
    assert "vision" in message.lower()


async def test_require_vision_support_allows_a_vision_model_and_plain_text() -> None:
    # A vision model may take the image...
    require_vision_support(_image_wire(), _model("v", "p", vision=True))
    # ...and a text-only turn is fine on any model (no raise either way).
    require_vision_support([{"role": "user", "content": "hi"}], _model("t", "p"))


# ---------------------------------------------------------------------------
# The real adapters refuse media on a text-only model - offline, before any client
# ---------------------------------------------------------------------------
async def _first_step_error(agen) -> str:
    """Drive an async-generator stream one step and return the error it raised."""
    message = ""
    try:
        await agen.__anext__()
    except RuntimeError as exc:
        message = str(exc)
    finally:
        await _aclose(agen)
    return message


async def _aclose(agen) -> None:
    close = getattr(agen, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:  # noqa: BLE001 - already finished; nothing to shut down
            pass


async def test_groq_adapter_refuses_media_on_a_text_only_model() -> None:
    from agent_native.models.groq_model import Groq

    # A dummy key so construction never touches the environment; the gate fires
    # before the client (or the key) is ever used, so this stays fully offline.
    stream = Groq(api_key="offline-unused").stream(_image_wire(), [], _model("g", "groq"))
    message = await _first_step_error(stream)
    assert "vision" in message.lower()


async def test_ollama_adapter_refuses_media_on_a_text_only_model() -> None:
    from agent_native.models.ollama_model import Ollama

    stream = Ollama().stream(_image_wire(), [], _model("o", "ollama"))
    message = await _first_step_error(stream)
    assert "vision" in message.lower()


# ---------------------------------------------------------------------------
# The plan's verify, end to end
# ---------------------------------------------------------------------------
async def test_vision_model_gets_a_grounded_answer() -> None:
    """Half one: an image reaches a vision-capable model and the answer is
    grounded in it (the model names the media type it actually received)."""
    seer = _VisionEcho()
    reg = ModelRegistry()
    reg.register_provider("vision", seer)
    reg.register_model("vision-1", _model("vision-1", "vision", vision=True))
    loop, db = _loop(reg)

    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    conv = Conversation(
        [
            system_message("You can see images.", session.id),
            user_message(session.id, "what is this?", media=[media_part(_PNG, "image/png")]),
        ]
    )
    result = await loop.run(conv, _context(session, AgentConfig(model="vision-1")))

    assert result.status is RunStatus.FINISHED
    assert seer.saw_image is True                 # the image was on the wire it received
    assert seer.seen_mime == "image/png"
    assert "image/png" in result.final_text       # ...and the answer is derived from it


async def test_text_only_model_reports_unsupported_rather_than_crashing() -> None:
    """Half two: a model that can't see images is handed one, and the run ends in a
    clean ERROR carrying a readable reason - loop.run returns, it does not raise."""
    gate = _GatingTextModel()
    reg = ModelRegistry()
    reg.register_provider("textonly", gate)
    reg.register_model("textonly-1", _model("textonly-1", "textonly"))  # vision False
    loop, db = _loop(reg)

    session = Session(agent="build", working_directory=".")
    await db.create_session(session)
    conv = Conversation(
        [
            system_message("Text only.", session.id),
            user_message(session.id, "what is this?", media=[media_part(_PNG, "image/png")]),
        ]
    )
    # Returns a result, never raises out of the loop - that's the "rather than crashing".
    result = await loop.run(conv, _context(session, AgentConfig(model="textonly-1")))

    assert result.status is RunStatus.ERROR
    assert "vision" in result.error.lower()
    assert gate.calls == 1                        # fired once; permanent error, so no retry


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_media_part_encodes_bytes,
        test_media_part_keeps_a_base64_string_as_is,
        test_document_media_is_not_an_image,
        test_user_message_carries_text_and_media,
        test_media_only_user_message_has_no_empty_text_part,
        test_render_text_only_user_is_still_a_plain_string,
        test_render_user_with_image_is_the_openai_content_list,
        test_render_image_only_omits_the_text_entry,
        test_render_passes_the_detail_hint_through,
        test_render_document_degrades_to_a_text_note,
        test_estimate_tokens_counts_an_image,
        test_ollama_splits_an_image_into_the_images_field,
        test_ollama_leaves_a_plain_text_turn_untouched,
        test_data_url_prefix_is_stripped,
        test_wire_has_media_detects_an_image_url,
        test_wire_has_media_is_false_for_text_only,
        test_wire_has_media_detects_the_ollama_images_field,
        test_require_vision_support_raises_for_a_text_only_model,
        test_require_vision_support_allows_a_vision_model_and_plain_text,
        test_groq_adapter_refuses_media_on_a_text_only_model,
        test_ollama_adapter_refuses_media_on_a_text_only_model,
        test_vision_model_gets_a_grounded_answer,
        test_text_only_model_reports_unsupported_rather_than_crashing,
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
        print("FAIL - multimodal:")
        for line in failures:
            print("  -", line)
        return 1
    print(
        f"PASS - multimodal: {len(tests)} tests (media_part, render shapes, Ollama "
        "split, wire detection, the gate, both adapters refusing, and the plan's "
        "two-half verify)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

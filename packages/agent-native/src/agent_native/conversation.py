"""The conversation: the one piece of state the whole agent is built around.

A run is just a conversation that grows. The model reads the whole thing, adds a
message (some text, maybe some tool calls), the tools add their results back as
more messages, and the model reads again. That is the entire loop.

`Conversation.render()` is the ONLY place that turns messages into the shape a
model wants to see. Keeping that in one method is deliberate: it is the thing that
is easy to get subtly wrong (a tool call with no matching result makes most models
error out), so there is exactly one place to get it right and one place to test.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.base import ToolFormat


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Small vocabularies
# ---------------------------------------------------------------------------
class Role(str, Enum):
    """Who a message came from."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PartType(str, Enum):
    """What a single piece of a message is."""

    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    COMPACTION = "compaction"
    MEDIA = "media"


class ToolCallStatus(str, Enum):
    """Where a tool call is in its life."""

    PENDING = "pending"                      # parsed, not looked at yet
    WAITING_PERMISSION = "waiting_permission"  # user is being asked
    DENIED = "denied"                        # policy or user said no
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


# ---------------------------------------------------------------------------
# The pieces a message is made of
# ---------------------------------------------------------------------------
class MessagePart:
    """Base for the things a message can contain. Not used directly."""

    part_type: PartType


@dataclass
class Text(MessagePart):
    """Plain text the user reads."""

    text: str
    part_type: PartType = field(default=PartType.TEXT, init=False)


@dataclass
class Reasoning(MessagePart):
    """The model thinking out loud. Hidden from the user by default."""

    text: str
    hidden: bool = True
    part_type: PartType = field(default=PartType.REASONING, init=False)


@dataclass
class ToolCall(MessagePart):
    """A request to run a tool, and later the result of running it.

    The same object is used for both sides of a tool call. On an ASSISTANT
    message it is the request (name + arguments). On a TOOL message it is the
    result (status + output/error). The id ties the two together, which is
    exactly what a model needs to match a result to its request.
    """

    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.PENDING
    output: str = ""
    error: str = ""
    part_type: PartType = field(default=PartType.TOOL_CALL, init=False)


@dataclass
class Compaction(MessagePart):
    """A short summary that stands in for a batch of older messages."""

    summary: str
    old_messages: list = field(default_factory=list)  # list[str] of message ids
    tokens_before: int = 0
    tokens_after: int = 0
    part_type: PartType = field(default=PartType.COMPACTION, init=False)


@dataclass
class Media(MessagePart):
    """An image or document handed to the model alongside text.

    The bytes are carried base64-encoded as a plain string, not as a file path or
    a `bytes` object, for one reason: a `Conversation` is persisted (to the
    database) and its events are JSON-serialized, so a part has to survive a
    round-trip through JSON. A path wouldn't (the file may be gone on resume) and
    raw bytes don't serialize. `media_part` does the encoding so a caller can hand
    over the raw bytes it has.

    `mime_type` is what tells the provider adapter how to label the payload
    (`image/png`, `image/jpeg`, `application/pdf`, ...). `detail` is the optional
    resolution hint some vision APIs read ("low"/"high"); "" means "no preference",
    the same empty-is-default convention `reasoning_effort` uses.
    """

    data: str                       # base64-encoded payload, no data: prefix
    mime_type: str = "image/png"
    detail: str = ""
    part_type: PartType = field(default=PartType.MEDIA, init=False)

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    def data_url(self) -> str:
        """The `data:<mime>;base64,<payload>` URL the OpenAI/Groq wire wants."""
        return f"data:{self.mime_type};base64,{self.data}"


@dataclass
class Usage:
    """How many tokens a model call cost.

    `reasoning_tokens` is a *breakdown*, not an addition: on the OpenAI-compatible
    APIs that report it, the tokens a reasoning model spends thinking are already
    counted inside `output_tokens` and billed as output. So it never enters
    `total_tokens` or the cost - it only says how much of the output was thinking,
    which is the thing a thinking-budget knob is there to make visible. A provider
    that doesn't break the number out (Ollama) leaves it at zero, honestly.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# A message
# ---------------------------------------------------------------------------
@dataclass
class Message:
    """One entry in the conversation. Made of one or more parts."""

    role: Role
    parts: list = field(default_factory=list)  # list[MessagePart]
    session_id: str = ""
    model: str = ""
    usage: Usage | None = None
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_now)

    # -- reading helpers -------------------------------------------------
    def text(self) -> str:
        """All the plain text in this message, joined."""
        return "".join(p.text for p in self.parts if isinstance(p, Text))

    def tool_calls(self) -> list:
        """The ToolCall parts in this message."""
        return [p for p in self.parts if isinstance(p, ToolCall)]

    def has_tool_calls(self) -> bool:
        return any(isinstance(p, ToolCall) for p in self.parts)


# -- message factories (so nobody hand-builds the fields) ----------------
def user_message(
    session_id: str, text: str = "", media: list | None = None
) -> Message:
    """A user turn: some text, some images/documents, or both.

    `media` is a list of `Media` parts (build them with `media_part`). Text stays
    a single `Text` part exactly as before, so every existing text-only caller is
    unchanged; media parts are appended after it. A media-only message carries no
    empty `Text` part, so `render` doesn't emit a blank text entry for it.
    """
    parts: list = [Text(text)] if (text or not media) else []
    parts.extend(media or [])
    return Message(role=Role.USER, session_id=session_id, parts=parts)


def media_part(
    data: bytes | bytearray | str, mime_type: str = "image/png", detail: str = ""
) -> Media:
    """A `Media` part from raw bytes (encoded here) or an already-base64 string.

    Callers almost always have the raw bytes of a file, not base64, so this does
    the encoding; passing a `str` treats it as already-encoded and stores it as-is.
    """
    import base64

    if isinstance(data, (bytes, bytearray)):
        payload = base64.b64encode(bytes(data)).decode("ascii")
    else:
        payload = data
    return Media(data=payload, mime_type=mime_type, detail=detail)


def system_message(text: str, session_id: str = "") -> Message:
    return Message(role=Role.SYSTEM, session_id=session_id, parts=[Text(text)])


def assistant_message(
    session_id: str,
    text: str = "",
    tool_calls: list | None = None,
    model: str = "",
    usage: Usage | None = None,
) -> Message:
    parts: list = []
    if text:
        parts.append(Text(text))
    if tool_calls:
        parts.extend(tool_calls)
    return Message(
        role=Role.ASSISTANT,
        session_id=session_id,
        parts=parts,
        model=model,
        usage=usage,
    )


def tool_result_message(session_id: str, result: ToolCall) -> Message:
    """A TOOL message carrying one finished tool call (status + output/error)."""
    return Message(role=Role.TOOL, session_id=session_id, parts=[result])


# ---------------------------------------------------------------------------
# The session and the conversation
# ---------------------------------------------------------------------------
@dataclass
class Session:
    """A single working thread: an agent, a folder to work in, a title."""

    agent: str = "build"
    title: str = ""
    working_directory: str = "."
    revision: int = 0
    id: str = field(default_factory=_new_id)


class Conversation:
    """The ordered list of messages, plus the one method that renders them."""

    def __init__(self, messages: list | None = None) -> None:
        self.messages: list = list(messages) if messages else []

    # -- growing it ------------------------------------------------------
    def add(self, message: Message) -> None:
        self.messages.append(message)

    def get_recent_messages(self, count: int) -> list:
        """The last `count` messages (used to protect recent turns from compaction)."""
        if count <= 0:
            return []
        return self.messages[-count:]

    def estimate_tokens(self) -> int:
        """A cheap guess at size. Providers can count exactly; this is for budgets."""
        total = 0
        for msg in self.messages:
            for part in msg.parts:
                if isinstance(part, (Text, Reasoning)):
                    total += _rough_tokens(part.text)
                elif isinstance(part, ToolCall):
                    total += _rough_tokens(part.name) + _rough_tokens(str(part.arguments))
                    total += _rough_tokens(part.output) + _rough_tokens(part.error)
                elif isinstance(part, Compaction):
                    total += _rough_tokens(part.summary)
                elif isinstance(part, Media):
                    # Media doesn't tokenize by character count - a vision model
                    # bills an image by how many tiles its resolution covers, which
                    # can't be known from the base64 here. A flat, deliberately
                    # conservative estimate keeps the compaction budget from ignoring
                    # images entirely; the real cost comes back on the USAGE event
                    # like every other exact number (see the Usage docstring).
                    total += _MEDIA_TOKEN_ESTIMATE
        return total

    # -- the invariant a model cares about -------------------------------
    def is_valid(self) -> bool:
        """True if every tool call the assistant made has a matching tool result.

        Most models reject a request where an assistant asked for a tool but no
        tool result with that id follows. This is the check that keeps the wire
        format legal; compaction must never break it.
        """
        answered: set = set()
        for msg in self.messages:
            if msg.role == Role.TOOL:
                for tc in msg.tool_calls():
                    answered.add(tc.id)
        for msg in self.messages:
            if msg.role == Role.ASSISTANT:
                for tc in msg.tool_calls():
                    if tc.id not in answered:
                        return False
        return True

    # -- the ONLY wire-format author -------------------------------------
    def render(self, tool_format: ToolFormat | None = None) -> list:
        """Turn the conversation into the list of messages a model expects.

        Only NATIVE (the OpenAI / Groq shape: an assistant message carries a
        `tool_calls` array, each result comes back as a `tool` message keyed by
        `tool_call_id`) is built here. TEXT / JSON tool formats - used by some
        local models that can't do native tool calls - are a later pass.
        """
        from .models.base import ToolFormat as _TF  # local import: avoid a cycle

        fmt = tool_format or _TF.NATIVE
        if fmt is not _TF.NATIVE:
            raise NotImplementedError(
                f"Only NATIVE tool format is rendered so far, not {fmt}. "
                "TEXT / JSON rendering for local models is a later pass."
            )

        wire: list = []
        for msg in self.messages:
            if msg.role == Role.SYSTEM:
                wire.append({"role": "system", "content": msg.text()})

            elif msg.role == Role.USER:
                # Plain string when it's only text - the shape every existing
                # message has and every test asserts. A list of typed parts only
                # when the turn carries media, which is the OpenAI/Groq multimodal
                # shape; the Ollama adapter translates that list to its own
                # `images` field (see ollama_model._to_ollama_messages).
                wire.append({"role": "user", "content": _user_content(msg)})

            elif msg.role == Role.ASSISTANT:
                entry: dict = {"role": "assistant"}
                text = msg.text()
                entry["content"] = text if text else None
                calls = msg.tool_calls()
                if calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": _json_args(tc.arguments),
                            },
                        }
                        for tc in calls
                    ]
                wire.append(entry)

            elif msg.role == Role.TOOL:
                # One wire "tool" message per finished tool call.
                for tc in msg.tool_calls():
                    wire.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tc.error if tc.status == ToolCallStatus.ERROR else tc.output,
                        }
                    )
        return wire


def _rough_tokens(text: str) -> int:
    """~4 characters per token. Deliberately crude; only used for budgeting."""
    if not text:
        return 0
    return max(1, len(text) // 4)


#: A flat per-image budgeting estimate, used only by `estimate_tokens`. Not a
#: bill: the exact cost is resolution-dependent and comes back on the USAGE event.
_MEDIA_TOKEN_ESTIMATE = 256


def _user_content(msg: Message) -> str | list:
    """The `content` field for a user message: a plain string, or a parts list.

    No media -> the joined text, exactly as before, so a text-only turn renders
    byte-for-byte what it always did. With media -> the OpenAI/Groq multimodal
    shape: a `text` part (only if there's text) followed by one `image_url` part
    per image, each carrying the base64 payload as a data URL. A non-image
    document has no portable content type across these providers, so it degrades
    to a labelled text note rather than a shape a provider would reject.
    """
    media = [p for p in msg.parts if isinstance(p, Media)]
    if not media:
        return msg.text()

    content: list = []
    text = msg.text()
    if text:
        content.append({"type": "text", "text": text})
    for m in media:
        if m.is_image:
            image_url: dict = {"url": m.data_url()}
            if m.detail:
                image_url["detail"] = m.detail
            content.append({"type": "image_url", "image_url": image_url})
        else:
            # Non-image document has no portable wire type; degrade to a labelled
            # text note so the turn is still sendable (matches test expectation).
            content.append({"type": "text", "text": f"Document ({m.mime_type}): {m.data[:200]}"})
    return content


def _json_args(arguments: dict) -> str:
    import json

    try:
        return json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"

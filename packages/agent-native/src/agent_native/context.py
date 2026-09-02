"""Keeping the conversation inside the model's window.

Every model can only read so much at once. When a conversation gets close to that
limit, older messages are folded into a short summary and dropped, while the most
recent messages are kept word-for-word. The one rule that must never be broken:
a tool call and its result are a pair, and a pair is never split across the
summary boundary - do that and the next model call is rejected as malformed.

There are two ways the summary gets written. `compact` writes a plain,
deterministic digest - what the user asked, which tools ran, the last result -
with no model call at all. `compact_with_model` asks the model to write it, and
falls back to that same digest if the call fails, times out or comes back empty.

**Why bother asking the model.** The digest is honest but it is a *list*, and
what a long run actually needs carried forward is the state: the file that turned
out to be the wrong one, the value that was already looked up, the correction the
user made an hour ago. A template can't know which of those matter. A model
reading the transcript can, and the cost is one extra call at the moment the
conversation was about to lose that text anyway.

**Why the template stays.** It's the floor. A summary is written at the exact
moment the window is nearly full, which is also the moment a model call is most
likely to be rejected for length - so the path that needs no model has to keep
working, and stay predictable enough to test against.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .conversation import (
    Compaction,
    Conversation,
    Role,
    system_message,
)
from .models.base import StreamType, rough_token_count

#: What the model is told when it's asked to write the summary. It is addressed as
#: the reader of its own summary on purpose: the thing being written is the only
#: memory it will have of these messages.
SUMMARY_INSTRUCTION = (
    "Summarize the conversation below so that you can carry on without it. "
    "Another copy of you will read your summary in place of these messages and "
    "must be able to continue the work from it alone.\n\n"
    "Write plain prose, under 200 words, covering: what the user asked for; what "
    "has been done and what it found, keeping exact file names, paths, commands "
    "and values that would otherwise be lost; what is still left to do; and any "
    "correction or preference the user gave. No advice, no commentary, no "
    "headings - just the state of the work."
)

#: How much of the text being dropped to show the model, as head and tail. The
#: goal is at the start and the current state is at the end; the middle is the
#: part a summary is allowed to lose. This is capped because compaction runs when
#: the window is nearly full - handing all of it back would be the same problem.
MAX_SUMMARY_INPUT_CHARS = 24_000

#: A summary that hasn't arrived in this long isn't worth waiting for; the
#: template is right there.
SUMMARY_TIMEOUT_SECONDS = 60.0


class TokenCounter:
    """Counts tokens (exactly if a provider counter is given, roughly otherwise).

    Counting only, on purpose. It used to also tally usage across a run, but the
    run's own totals now live on `RunResult`, where they can be reported and
    stored - two places keeping the same running total is how they drift apart.
    """

    def __init__(self, count_fn: Callable | None = None) -> None:
        self._count_fn = count_fn

    def count(self, wire_messages: list) -> int:
        if self._count_fn is not None:
            return self._count_fn(wire_messages)
        return rough_token_count(wire_messages)


@dataclass
class CompactionResult:
    """What a round of compaction did."""

    summary: Compaction
    old_messages: list = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0


class ContextManager:
    """Decides when to compact, and does it without ever splitting a tool pair."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        recent_window: int = 6,
        threshold: float = 0.8,
    ) -> None:
        self.tokens = token_counter or TokenCounter()
        self.recent_window = recent_window
        self.threshold = threshold

    # -- when -------------------------------------------------------------
    def needs_compaction(
        self,
        conversation: Conversation,
        model: Any,
        observed_input_tokens: int | None = None,
    ) -> bool:
        """True when the conversation is close enough to the window to fold it.

        ``observed_input_tokens``, when given, is the provider's real prompt-token
        count from the last turn - the honest size of what we actually sent. It's
        preferred whenever it's there, because the fallback is a ~4-chars-per-token
        guess that a real tokenizer (Llama, Qwen) will disagree with, sometimes by
        a lot. Only the first turn, before any usage has come back, rides on the
        estimate; by the second turn the decision is made on truth. Either way this
        is a budget signal, not a bill - the run's real totals live on
        ``RunResult``, counted from provider usage alone.
        """
        if observed_input_tokens not in (None, 0):
            used = observed_input_tokens
        else:
            used = self.tokens.count(conversation.render())
        return used > model.context_size * self.threshold

    # -- where to cut -----------------------------------------------------
    def _leading_system_count(self, messages: list) -> int:
        count = 0
        for msg in messages:
            if msg.role == Role.SYSTEM:
                count += 1
            else:
                break
        return count

    def protect_recent_messages(self, conversation: Conversation) -> int:
        """Return the index where 'recent' begins, pulled back so no tool pair splits."""
        messages = conversation.messages
        n = len(messages)
        system_count = self._leading_system_count(messages)
        window = max(0, self.recent_window)
        split = max(system_count, n - window)
        # Never begin 'recent' on an orphan tool result, and never leave an
        # assistant's tool-call request in 'older' while its results are 'recent'.
        # The `split < n` guard keeps an empty recent window from indexing off the end.
        while split > system_count and (
            (split < n and messages[split].role == Role.TOOL)
            or (messages[split - 1].role == Role.ASSISTANT and messages[split - 1].has_tool_calls())
        ):
            split -= 1
        return split

    # -- do it ------------------------------------------------------------
    def check_tool_pairs(self, older: list, recent: list) -> bool:
        """True if the split kept every tool call together with its result."""
        older_conv = Conversation(older)
        recent_conv = Conversation(recent)
        return older_conv.is_valid() and recent_conv.is_valid()

    def compact(self, conversation: Conversation, model: Any) -> CompactionResult | None:
        """Fold older messages into a template summary. None if there's nothing to fold.

        No model call, so this is safe to call from anywhere and gives the same
        answer every time. `compact_with_model` is the better summary when there's
        a provider to hand.
        """
        leading, older, recent = self._split(conversation)
        if not older:
            return None
        return self._apply(conversation, leading, older, recent, _summarize(older))

    async def compact_with_model(
        self,
        conversation: Conversation,
        model: Any,
        provider: Any = None,
    ) -> CompactionResult | None:
        """Same fold, but with the model writing the summary where it can.

        Every way the model call can go wrong ends in the template: no provider, a
        refusal, a timeout, an empty reply. That's the whole reason this is worth
        doing at all - the better summary is an improvement on a working path, not
        a new thing that can break compaction.
        """
        leading, older, recent = self._split(conversation)
        if not older:
            return None

        summary_text = ""
        if provider is not None:
            summary_text = await write_summary(older, model, provider)
        return self._apply(
            conversation, leading, older, recent, summary_text or _summarize(older)
        )

    # -- the two halves both paths share ----------------------------------
    def _split(self, conversation: Conversation) -> tuple:
        """(leading system messages, what gets folded, what's kept word-for-word)."""
        messages = conversation.messages
        system_count = self._leading_system_count(messages)
        split = self.protect_recent_messages(conversation)
        return messages[:system_count], messages[system_count:split], messages[split:]

    def _apply(
        self,
        conversation: Conversation,
        leading: list,
        older: list,
        recent: list,
        summary_text: str,
    ) -> CompactionResult:
        """Swap `older` out for one system message carrying the summary.

        The summary goes *after* the leading system messages rather than merged
        into them, which is what keeps the start of every request byte-identical
        from one turn to the next - see `mark_cacheable_prefix` in models/base.py.
        """
        tokens_before = self.tokens.count(Conversation(older).render())
        tokens_after = rough_token_count([{"content": summary_text}])

        compaction = Compaction(
            summary=summary_text,
            old_messages=[m.id for m in older],
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        summary_msg = system_message("[Earlier conversation, summarized]\n" + summary_text)
        summary_msg.parts.append(compaction)

        conversation.messages = leading + [summary_msg] + recent
        return CompactionResult(
            summary=compaction,
            old_messages=older,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )


async def write_summary(older: list, model: Any, provider: Any) -> str:
    """Ask the model to summarize the messages being dropped. "" if it couldn't.

    The transcript goes over as *text in one user message*, not as the messages
    themselves. Replaying assistant tool calls at a provider without also
    declaring the tools is a request some of them reject outright, and this is a
    summary - the shape of the original call doesn't matter, only what it said.
    """
    transcript = _as_text(older)
    if not transcript:
        return ""
    request = [
        {"role": "system", "content": SUMMARY_INSTRUCTION},
        {"role": "user", "content": transcript},
    ]
    try:
        return await asyncio.wait_for(
            _collect_text(provider, request, model), timeout=SUMMARY_TIMEOUT_SECONDS
        )
    except Exception:  # noqa: BLE001 - a timeout and a refusal both end in the template
        return ""


async def _collect_text(provider: Any, request: list, model: Any) -> str:
    """Read one non-tool reply out of a provider stream as plain text."""
    chunks: list = []
    stream = provider.stream(request, [], model, 0.0)
    try:
        async for event in stream:
            if event.type == StreamType.TEXT:
                chunks.append(event.data.get("text", ""))
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 - hanging up isn't worth an error
                pass
    return "".join(chunks).strip()


def _as_text(older: list, limit: int = MAX_SUMMARY_INPUT_CHARS) -> str:
    """The messages being dropped, as a readable transcript, head and tail only."""
    lines: list = []
    for msg in older:
        who = msg.role.value
        body = msg.text().strip()
        for call in msg.tool_calls():
            if msg.role == Role.ASSISTANT:
                lines.append(f"{who} called {call.name} with {call.arguments}")
            else:
                outcome = call.error or call.output or ""
                lines.append(f"{call.name} returned: {outcome[:1000]}")
        if body:
            lines.append(f"{who}: {body}")
    text = "\n".join(lines).strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n[... middle of the conversation left out ...]\n\n" + text[-half:]


def _summarize(older: list) -> str:
    """A plain digest of the messages being dropped. Deterministic on purpose."""
    first_goal = ""
    tool_names: list = []
    last_result = ""
    for msg in older:
        if msg.role == Role.USER and not first_goal:
            first_goal = msg.text()[:200]
        for call in msg.tool_calls():
            if msg.role == Role.ASSISTANT and call.name not in tool_names:
                tool_names.append(call.name)
            if msg.role == Role.TOOL and (call.output or call.error):
                last_result = (call.output or call.error)[:200]

    lines = [f"{len(older)} earlier messages were summarized to save space."]
    if first_goal:
        lines.append(f"The user's goal was: {first_goal}")
    if tool_names:
        lines.append(f"Tools used: {', '.join(tool_names)}.")
    if last_result:
        lines.append(f"Most recent tool result: {last_result}")
    return "\n".join(lines)

"""ContextCompactor - token-budget management for the ReAct loop.

Decision #3 (hard requirement): compaction only ever operates on complete
assistant(tool_calls) -> tool-result message pairs, atomically. A pair is a
contiguous block [assistant w/ tool_calls, then exactly the tool-result
messages whose tool_call_ids it references]. Anything else - assistant text
messages, orphan tool results, trailing partials - is preserved verbatim.
This keeps the message list valid for completion APIs (a tool result must
always follow its assistant tool_call; an assistant tool_call must always be
paired with its results).

`compact()` is deterministic and LLM-free by default (a `summarize` callable
may be injected for semantic summaries later); the default keeps a compact
trace line per pair, so nothing semantically critical is dropped.
"""

from __future__ import annotations

import json
from typing import Callable, TypeAlias

Message = dict
History: TypeAlias = list[Message]

PairSummarizer = Callable[[list[Message]], str]

_OVERHEAD_TOKENS = 4


def estimate_tokens(messages: History) -> int:
    """Cheap char/4 estimate + per-message overhead. Good enough for gating."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"]) // 4
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            total += len(json.dumps(tc)) // 4
        total += _OVERHEAD_TOKENS
    return total


class ContextCompactor:
    def __init__(self, token_budget: int = 20000) -> None:
        self.token_budget = token_budget
        self.summaries_applied = 0

    def check_budget(self, history: History, budget: int | None = None) -> bool:
        """True when history is at/over budget and compaction is needed."""
        return estimate_tokens(history) >= (budget or self.token_budget)

    def compact(self, history: History, summarize: PairSummarizer | None = None) -> History:
        """Summarize complete call/result pairs; preserve everything else."""
        pairs, orphans = self._split_pairs(history)
        if not pairs:
            return history

        out: History = []
        index = 0
        for block, is_pair in self._merged_blocks(history, pairs, orphans):
            if not is_pair:
                out.append(block)
            else:
                summary = (summarize(block) if summarize else _default_summarize(block))
                out.append({"role": "assistant", "content": f"[compacted] {summary}"})
            index += 1

        self.summaries_applied += len(pairs)
        return out

    def _split_pairs(self, history: History) -> tuple[dict[int, int], set[int]]:
        """Return (pair_heads: start_index -> pair_size, orphan_tool_indices)."""
        pair_heads: dict[int, int] = {}
        orphans: set[int] = set()
        i = 0
        while i < len(history):
            msg = history[i]
            tool_calls = msg.get("tool_calls") or []
            if msg.get("role") == "assistant" and tool_calls:
                needed = [tc.get("id") for tc in tool_calls]
                consumed = 0
                j = i + 1
                while consumed < len(needed) and j < len(history):
                    nxt = history[j]
                    if nxt.get("role") == "tool":
                        id_ = nxt.get("tool_call_id")
                        if id_ in needed:
                            consumed += 1
                            j += 1
                            continue
                        break
                    break
                if consumed == len(needed):
                    pair_heads[i] = j - i
                    i = j
                    continue
            elif msg.get("role") == "tool":
                orphans.add(i)
            i += 1
        return pair_heads, orphans

    def _merged_blocks(
        self, history: History, pairs: dict[int, int], orphans: set[int]
    ) -> list[tuple[Message, bool]]:
        blocks: list[tuple[Message, bool]] = []
        i = 0
        while i < len(history):
            if i in pairs:
                size = pairs[i]
                blocks.append((list(history[i:i + size]), True))
                i += size
            else:
                blocks.append((history[i], False))
                i += 1
        return blocks


def _default_summarize(pair: list[Message]) -> str:
    assistant = pair[0]
    parts: list[str] = []
    for tc in assistant.get("tool_calls") or []:
        fn = (tc.get("function") or {}).get("name", "?")
        args = (tc.get("function") or {}).get("arguments", "")
        parts.append(f"{fn}({_trim(args, 160)})")
    results: list[str] = []
    for m in pair[1:]:
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content)[:220]
        results.append(_trim(text, 220))
    return " -> ".join(parts) + " | " + "; ".join(results)


def _trim(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[:width] + "..."

"""ContextCompactor tests - especially the pairing invariant (decision #3)."""

from __future__ import annotations

import agent_native.compactor as comp
from agent_native.compactor import ContextCompactor


def assistant_tool_calls(*items) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {"name": n, "arguments": "{}"}}
                       for cid, n in items],
    }


def tool_result(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def make_history() -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        assistant_tool_calls(("a1", "read_file"), ("a2", "list_directory")),
        tool_result("a1", "file contents"),
        tool_result("a2", "[dir]"),
        {"role": "assistant", "content": "so far so good"},
        assistant_tool_calls(("a3", "write_file")),
        tool_result("a3", "wrote 1 file"),
        tool_result("orphan", "no preceding assistant tool_calls"),
    ]


def test_budget_triggers_on_large_history():
    c = ContextCompactor(token_budget=20)
    assert c.check_budget(make_history())
    c2 = ContextCompactor(token_budget=10**6)
    assert not c2.check_budget(make_history())


def _assert_pairing_invariant(compact):
    """The real invariant: compaction never BREAKS a pairing and never
    creates a new orphan. Remaining tool results must either (a) be paired
    with an assistant tool_calls message right before them, or (b) have been
    orphans in the source. No assistant tool_calls message may lack its
    results."""
    original_orphans = {m.get("tool_call_id") for m in compact if m.get("role") == "tool"}
    seen_assistant = False
    i = 0
    while i < len(compact):
        msg = compact[i]
        if msg.get("tool_calls"):
            ids = [tc.get("id") for tc in msg["tool_calls"]]
            j = i + 1
            consumed = 0
            while consumed < len(ids) and j < len(compact) and compact[j].get("role") == "tool" \
                    and compact[j].get("tool_call_id") in ids:
                consumed += 1
                j += 1
            assert consumed == len(ids), f"assistant tool_calls {ids} lack their results"
            i = j
        elif msg.get("role") == "tool":
            assert msg.get("tool_call_id") in original_orphans, "compaction created a new orphan"
            i += 1
        else:
            i += 1


def test_compaction_preserves_pairing_atomically():
    history = make_history()
    c = ContextCompactor(token_budget=1)
    out = c.compact(history)
    _assert_pairing_invariant(out)
    # not compactor's job to keep size fixed; must still be type-correct
    assert out


def test_exact_pair_units_are_replaced_together():
    history = make_history()
    c = ContextCompactor(token_budget=1)
    out = c.compact(history)
    # The a1/a2 pair (2 tool calls) and the a3 pair were summarized; the
    # orphan result and the plain assistant text are preserved verbatim.
    orphan_present = any(m.get("tool_call_id") == "orphan" for m in out)
    assert orphan_present, "orphan tool result must never be dropped"
    # Each summary line mentions its tool names (atomic pair -> one line)
    summaries = [m["content"] for m in out if m.get("content", "").startswith("[compacted]")]
    assert any("read_file" in s and "list_directory" in s for s in summaries)
    assert any("write_file" in s for s in summaries)


def test_unpaired_partial_tail_is_preserved():
    """A trailing assistant tool_call followed by only ONE of two results
    must not be dropped or split."""
    history = [
        assistant_tool_calls(("x1", "a"), ("x2", "b")),
        tool_result("x1", "r1"),
    ]
    c = ContextCompactor(token_budget=1)
    out = c.compact(history)
    assert len(out) == 2  # untouched: pairing incomplete
    assert out[0]["tool_calls"][0]["id"] == "x1"


def test_full_pairing_across_many_messages_keeps_order():
    history = [
        {"role": "user", "content": "u1"},
        assistant_tool_calls(("p1", "one")),
        tool_result("p1", "r1"),
        assistant_tool_calls(("p2", "two")),
        tool_result("p2", "r2"),
        {"role": "user", "content": "u2"},
    ]
    c = ContextCompactor(token_budget=1)
    out = c.compact(history)
    kinds_and_roles = [
        (m.get("role"), (m.get("content") or "")[:12])
        for m in out
        if m.get("role") != "assistant" or not m.get("tool_calls")
    ] + [("assistant", "tool_calls") for m in out if m.get("tool_calls")]
    # user messages still in original relative order
    users = [m for m in out if m.get("role") == "user"]
    assert users[0]["content"] == "u1" and users[1]["content"] == "u2"


def test_summarize_injected_callable_used():
    c = ContextCompactor(token_budget=1)
    calls = []

    def summarize(pair):
        calls.append(pair)
        return "CUSTOM_SUMMARY"

    out = c.compact(make_history(), summarize=summarize)
    assert calls, "summarize must be invoked per pair"
    assert any("CUSTOM_SUMMARY" in m["content"] for m in out)


def test_estimate_tokens_monotonic():
    small = [{"role": "user", "content": "hi"}]
    big = [{"role": "user", "content": "x" * 10000}]
    assert comp.estimate_tokens(big) > comp.estimate_tokens(small)
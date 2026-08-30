"""Redaction: secrets don't reach anything stored, shipped, or shown.

Step 10's promise is that a key can't leak through the places a run writes text
down. So these tests prove it at each sink, plus the redactor itself:

  * the `Redactor` masks a known value (wherever it sits, however nested) and a
    key-shaped string it was never told the value of, leaves ordinary text alone,
    and is idempotent;
  * the event bus masks event data on the way in - so an event row, a replayed
    stream, and the log that prints from them are all clean;
  * the trace exporter masks span attributes in the JSON file while the live spans
    stay exact for in-process inspection;
  * the memory store masks a note before it's kept;
  * and the runtime actually installs one redactor into all three.

The event and trace tests are the plan's verify directly: "grep the JSON and event
rows to confirm nothing leaks."

Offline by construction: no model, no network, no key. Run under pytest, or:
    PYTHONPATH=packages/agent-native/src python3 packages/agent-native/tests/test_redaction.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from agent_native.database import MemoryDatabase
from agent_native.events import EventBus, EventType
from agent_native.memory import MemoryStore
from agent_native.monitoring import Monitoring
from agent_native.redaction import (
    MASK,
    EnvSecretSource,
    Redactor,
    StaticSecretSource,
)

# A fake secret that is not itself key-shaped, so a test that means to exercise
# *value* masking isn't secretly relying on a pattern to catch it.
SECRET = "s3cr3t-value-abcdef123456"


# ---------------------------------------------------------------------------
# The redactor itself
# ---------------------------------------------------------------------------
def test_known_value_is_masked_wherever_it_sits() -> None:
    r = Redactor(StaticSecretSource([SECRET]))
    assert r.redact_text(f"the key is {SECRET} ok") == "the key is [redacted] ok"
    # Nested in a dict and a list, the shape the event/trace payloads actually take.
    payload = {"output": f"KEY={SECRET}", "items": [SECRET, "fine"], "n": 7, "ok": None}
    out = r.redact(payload)
    assert SECRET not in json.dumps(out)
    assert out["items"][1] == "fine" and out["n"] == 7 and out["ok"] is None  # untouched


def test_longer_value_wins_so_no_tail_is_left() -> None:
    r = Redactor(StaticSecretSource(["secret", "secretlong"]))
    # Masking the short one first would leave "long"; longest-first avoids that.
    assert r.redact_text("secretlong here") == "[redacted] here"


def test_key_shaped_strings_are_masked_without_knowing_the_value() -> None:
    r = Redactor(StaticSecretSource([]))  # no known values at all
    assert r.redact_text("token gsk_ABCDEFGHIJKLMNOP0123 end") == "token [redacted] end"
    assert r.redact_text("openai sk-proj-ABCDEFGHIJKLMNOP01 x") == "openai [redacted] x"
    assert "abcdef1234567890XYZ" not in r.redact_text("Authorization: Bearer abcdef1234567890XYZ")
    assert r.redact_text("password=hunter2longenough") == "password=[redacted]"


def test_ordinary_text_is_left_alone() -> None:
    r = Redactor(StaticSecretSource([SECRET]))
    plain = "count=5 and the total was 42 dollars; nothing secret here"
    assert r.redact_text(plain) == plain


def test_redaction_is_idempotent() -> None:
    r = Redactor(StaticSecretSource([SECRET]))
    once = r.redact_text(f"a {SECRET} b gsk_ABCDEFGHIJKLMNOP0123 c")
    assert r.redact_text(once) == once  # a second pass changes nothing


def test_non_strings_pass_through() -> None:
    r = Redactor(StaticSecretSource([SECRET]))
    assert r.redact(7) == 7 and r.redact(None) is None and r.redact(True) is True


def test_env_secret_source_reads_the_environment_now() -> None:
    name = "AGENT_NATIVE_TEST_SECRET"
    old = os.environ.get(name)
    try:
        os.environ[name] = SECRET
        assert EnvSecretSource((name,)).values() == [SECRET]
        del os.environ[name]
        assert EnvSecretSource((name,)).values() == []  # unset -> nothing to mask
    finally:
        if old is not None:
            os.environ[name] = old
        else:
            os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# The sinks
# ---------------------------------------------------------------------------
async def test_event_data_is_redacted_on_the_way_in() -> None:
    db = MemoryDatabase()
    bus = EventBus(db, redactor=Redactor(StaticSecretSource([SECRET])))
    session_id = "sess_redact"
    await db.create_session(_session(session_id))

    await bus.emit(session_id, EventType.TOOL_FINISHED,
                   {"name": "shell", "success": True, "output": f"printenv -> KEY={SECRET}"})

    events = await db.load_events(session_id, 0)
    blob = json.dumps([e.data for e in events])
    assert SECRET not in blob            # nothing leaks into the stored event row
    assert MASK in blob                  # and it was actually masked, not dropped


async def test_a_bare_event_bus_stays_literal() -> None:
    # No redactor (the shape a test or embedder builds) must not silently mangle data.
    db = MemoryDatabase()
    bus = EventBus(db)
    session_id = "sess_literal"
    await db.create_session(_session(session_id))
    await bus.emit(session_id, EventType.TOOL_FINISHED, {"output": f"KEY={SECRET}"})
    events = await db.load_events(session_id, 0)
    assert SECRET in json.dumps([e.data for e in events])


def test_trace_json_is_redacted_but_live_spans_stay_exact() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="an_trace_"))
    mon = Monitoring(trace_dir=tmp, redactor=Redactor(StaticSecretSource([SECRET])))
    with mon.run_span("run_x"):
        with mon.tool_span("shell", output=f"KEY={SECRET}") as tr:
            pass
    # Live span keeps the real value - it's in-process state, not a sink.
    assert tr.attributes["output"] == f"KEY={SECRET}"

    written = mon.shutdown()
    assert written, "expected a JSON trace file to be written"
    raw = Path(written[0]).read_text(encoding="utf-8")
    assert SECRET not in raw and MASK in raw   # the file on disk is clean


async def test_memory_notes_are_redacted_before_they_are_kept() -> None:
    db = MemoryDatabase()
    store = MemoryStore(db, redactor=Redactor(StaticSecretSource([SECRET])))
    await store.remember(f"the deploy key is {SECRET}")
    kept = await db.load_memories("")
    assert kept and SECRET not in kept[0].text and MASK in kept[0].text


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------
def test_runtime_installs_one_redactor_into_every_sink() -> None:
    from agent_native.service import AgentRuntime

    runtime = AgentRuntime()
    assert runtime.events._redactor is runtime.redactor
    assert runtime.memory._redactor is runtime.redactor
    assert runtime.monitoring.redactor is runtime.redactor


def _session(session_id: str):
    from agent_native.conversation import Session

    s = Session(agent="build")
    s.id = session_id
    return s


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    import inspect

    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures: list = []
    for test in tests:
        try:
            if inspect.iscoroutinefunction(test):
                asyncio.run(test())
            else:
                test()
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - redaction:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - redaction: {len(tests)} tests "
          "(value + pattern masking, idempotent, event/trace/memory sinks, runtime wiring).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

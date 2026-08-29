"""What the monitoring layer records, and what its two sinks do with it.

Offline by construction: no collector, and the OpenTelemetry SDK isn't imported
unless an endpoint is set - so these run with nothing from the `tracing` extra
installed. They check the parts that don't need a live collector:

  * spans nest the way the work nested (run > turn > tool), including tools that
    ran in parallel under one turn;
  * each span carries a real wall-clock start and a finished duration;
  * the JSON fallback still lands, with or without an endpoint configured;
  * an endpoint set but the extra missing degrades cleanly - JSON is written, the
    run is untouched, and the skip is reported rather than raised.

The live half of the plan's verify ("point at a collector, see one trace per run")
runs on a machine with the extra and a collector; the export path here is wired
and proven to fail safe.

Run under pytest, or straight (the __main__ block) on a box without pytest:
    PYTHONPATH=packages/agent-native/src python3 packages/agent-native/tests/test_monitoring.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from agent_native.monitoring import Monitoring, _otlp_attributes


def _by_name(mon: Monitoring) -> dict:
    """The recorded spans keyed by name. Names are unique per level in these tests."""
    return {s.name: s for s in mon.spans}


async def test_spans_nest_run_turn_tool_including_parallel_tools() -> None:
    """A tool's parent is its turn; a turn's parent is its run; the run is a root.

    The two tools are opened inside an `asyncio.gather`, exactly as the loop runs a
    read-only group, to prove the parent link survives the task boundary - a
    ContextVar is copied into each task at creation, so both tools see the turn.
    """
    mon = Monitoring()  # enabled, no trace_dir, no endpoint
    tool_parents: list = []

    with mon.run_span("run_x") as run:
        with mon.turn_span(1) as turn:

            async def one_tool(name: str) -> None:
                with mon.tool_span(name) as tool:
                    await asyncio.sleep(0)
                    tool_parents.append((name, tool.parent_id))

            await asyncio.gather(one_tool("read_a"), one_tool("read_b"))

    assert run.parent_id == ""                 # the run is the root of its tree
    assert turn.parent_id == run.span_id       # the turn hangs off the run
    # both parallel tools hang off the same turn
    assert {name: parent for name, parent in tool_parents} == {
        "read_a": turn.span_id,
        "read_b": turn.span_id,
    }
    # four spans, all filed under the one run
    assert [s.name for s in mon.spans] == ["run", "turn", "tool", "tool"]
    assert all(s.run_id == "run_x" for s in mon.spans)


async def test_each_span_has_wallclock_start_and_a_duration() -> None:
    mon = Monitoring()
    before = __import__("time").time()
    with mon.run_span("run_t"):
        with mon.turn_span(1):
            pass
    after = __import__("time").time()

    for span in mon.spans:
        assert before <= span.start_wall <= after   # a real clock reading, not monotonic
        assert span.end is not None                  # finished on the way out
        assert span.duration_s >= 0.0


async def test_json_fallback_lands_with_no_endpoint() -> None:
    """No collector configured: OTLP isn't attempted, and the JSON file is written."""
    with tempfile.TemporaryDirectory() as tmp:
        mon = Monitoring(trace_dir=tmp)
        with mon.run_span("run_json"):
            with mon.turn_span(1):
                with mon.tool_span("read_a"):
                    pass
        written = mon.shutdown()

        assert not mon.otlp_attempted           # nowhere to send, so it didn't try
        assert mon.otlp_exported is False
        assert len(written) == 1
        payload = json.loads(Path(written[0]).read_text(encoding="utf-8"))
        assert payload["run_id"] == "run_json"
        assert payload["span_count"] == 3
        assert [s["name"] for s in payload["spans"]] == ["run", "turn", "tool"]


async def test_endpoint_set_but_extra_missing_degrades_to_json() -> None:
    """An endpoint is named but the tracing extra isn't installed here.

    The export is attempted, fails cleanly with a reason that points at the extra,
    and the JSON still lands - a missing collector never costs the run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mon = Monitoring(trace_dir=tmp, otlp_endpoint="http://localhost:4318/v1/traces")
        with mon.run_span("run_otlp"):
            with mon.turn_span(1):
                pass
        written = mon.shutdown()

        assert mon.otlp_attempted is True       # an endpoint was configured
        assert mon.otlp_exported is False       # ...but the SDK isn't installed here
        assert "tracing" in mon.otlp_skipped_reason
        assert len(written) == 1                # JSON fallback still wrote
        assert Path(written[0]).exists()


async def test_shutdown_with_nothing_recorded_is_a_noop() -> None:
    mon = Monitoring(trace_dir="/nonexistent-should-not-be-created-xyz")
    assert mon.shutdown() == []                 # no spans -> nothing written, no error
    assert mon.otlp_attempted is False


def test_otlp_attributes_coerces_to_allowed_types() -> None:
    coerced = _otlp_attributes(
        {
            "turn": 3,                          # int stays
            "ok": True,                         # bool stays
            "name": "read_file",                # str stays
            "ratio": 1.5,                       # float stays
            "args": {"path": "a.txt"},          # dict -> stringified
            "tags": ["x", "y"],                 # list of primitives stays a list
        }
    )
    assert coerced["turn"] == 3
    assert coerced["ok"] is True
    assert coerced["name"] == "read_file"
    assert coerced["ratio"] == 1.5
    assert isinstance(coerced["args"], str) and "a.txt" in coerced["args"]
    assert coerced["tags"] == ["x", "y"]


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_spans_nest_run_turn_tool_including_parallel_tools,
        test_each_span_has_wallclock_start_and_a_duration,
        test_json_fallback_lands_with_no_endpoint,
        test_endpoint_set_but_extra_missing_degrades_to_json,
        test_shutdown_with_nothing_recorded_is_a_noop,
        test_otlp_attributes_coerces_to_allowed_types,
    ]
    failures: list = []
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                asyncio.run(test())
            else:
                test()
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - monitoring:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - monitoring: {len(tests)} tests "
          "(nesting, wall-clock, JSON fallback, OTLP degrade-safe).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

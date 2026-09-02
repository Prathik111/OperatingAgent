"""Monitoring: one trace per run, so a slow or failed run can be inspected later.

A span is opened for the run, for each turn inside it, and for each tool call
inside that. On shutdown the spans are written two ways: pushed to an
OpenTelemetry collector if one is configured, and always dropped as one JSON file
per run on disk. Either way they answer the question this exists to answer:
*where did the time go?* Usually the answer is one tool call, and you can see it
at a glance.

**The OTLP sink and the JSON fallback are independent.** The collector is the
primary sink when native-vs-LangGraph numbers have to line up across runs in one
place; the JSON is what lands on a laptop with no collector at all. So an endpoint
being set turns OTLP on, a `trace_dir` being set turns JSON on, and neither
depends on the other. `shutdown()` still returns the JSON paths, because that is
what the CLI prints.

Two deliberately small choices carry over from before. The OpenTelemetry SDK is
imported *lazily*, inside `shutdown()` and only when an endpoint is configured, so
the package still imports with nothing from the `tracing` extra installed. And
recording is on by default while *writing* is not: a library that starts leaving
files in someone's folder because they imported it is a bad neighbour, so a file
is only written when a caller says where to put it, and a span is only shipped
when a caller names a collector.

The span the loop records is a lightweight `Trace`, not a live OTel span - the hot
path stays dependency-free. The conversion to real OTel spans happens once, at
shutdown, by replaying the recorded traces with their parent links and wall-clock
timestamps. That replay is why `Trace` carries a `span_id`, a `parent_id` and a
`start_wall`: enough to rebuild the run > turn > tool tree and place it in real
time, without holding the SDK open for the length of a run.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_AUTO_LANGFUSE = object()

#: Which run the code currently executing belongs to. A ContextVar rather than a
#: plain attribute because two runs can be in flight in the same process, and
#: each one's turns and tools have to be filed under the right run.
_CURRENT_RUN: ContextVar[str] = ContextVar("agent_native_run_id", default="")

#: The span the code currently executing sits inside, by id. This is what gives a
#: turn its run as a parent and a tool its turn, so the exported trace nests the
#: way the work actually nested. A ContextVar for the same reason as the run id -
#: and because `asyncio.gather` copies the current context into each task it
#: starts, parallel tool spans each inherit the right turn as their parent for
#: free, with no locking.
_CURRENT_SPAN: ContextVar[str] = ContextVar("agent_native_span_id", default="")


@dataclass
class Trace:
    """One span: a named, timed slice of work with some attributes attached.

    `start` is a monotonic reading - right for measuring a duration, meaningless as
    a wall-clock time - so `start_wall` is captured alongside it for the one place
    that needs real time: an OTLP span carries an absolute start and end. The
    duration comes from the monotonic pair; the placement in time comes from
    `start_wall`. `span_id`/`parent_id` record where this span sits in the run's
    tree so the export can rebuild it.
    """

    name: str
    attributes: dict = field(default_factory=dict)
    start: float = field(default_factory=time.monotonic)
    end: float | None = None
    run_id: str = ""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str = ""
    start_wall: float = field(default_factory=time.time)

    def set(self, **attributes) -> None:
        self.attributes.update(attributes)

    def finish(self) -> None:
        if self.end is None:
            self.end = time.monotonic()

    @property
    def duration_s(self) -> float:
        return (self.end or time.monotonic()) - self.start


class Monitoring:
    """Opens spans for runs, turns and tools. Records them; ships them on shutdown."""

    def __init__(
        self,
        enabled: bool = True,
        trace_dir: Path | str | None = None,
        max_spans: int = 10_000,
        *,
        otlp_endpoint: str = "",
        otlp_headers: dict | None = None,
        service_name: str = "agent-native",
        redactor: Any | None = None,
        langfuse_client: Any = _AUTO_LANGFUSE,
    ) -> None:
        self.enabled = enabled
        # No directory means "remember, don't write". Tests use that; so does any
        # caller who wants the spans in memory and nothing on disk.
        self.trace_dir = Path(trace_dir).expanduser() if trace_dir else None
        self.max_spans = max_spans
        # An explicit endpoint wins; otherwise the standard OTEL_* environment
        # variables decide, read by the exporter itself at shutdown. Empty here and
        # unset there means the OTLP path is simply never taken.
        self.otlp_endpoint = otlp_endpoint
        self.otlp_headers = otlp_headers or {}
        self.service_name = service_name
        # Applied to span attributes at export only, never to the live spans - so a
        # secret that reached an attribute can't land in the JSON file or the OTLP
        # collector, while `self.spans` stays exact for in-process inspection. None
        # means no redaction (the runtime installs one; a bare Monitoring stays literal).
        self.redactor = redactor
        self.langfuse_client = self._resolve_langfuse_client(langfuse_client)
        self.langfuse_trace_ids: dict[str, str] = {}
        self.spans: list = []
        self.written: list = []  # paths written by the most recent shutdown()
        # What the most recent shutdown did with the OTLP sink, so a CLI or a test
        # can tell "shipped" from "no collector configured" from "the extra isn't
        # installed" without guessing.
        self.otlp_attempted = False
        self.otlp_exported = False
        self.otlp_skipped_reason = ""

    @staticmethod
    def _resolve_langfuse_client(client: Any) -> Any | None:
        if client is not _AUTO_LANGFUSE:
            return client
        try:
            from observability import get_client

            return get_client()
        except Exception:
            return None

    @contextmanager
    def _langfuse_observation(self, trace: Trace) -> Iterator[Any | None]:
        if not self.enabled:
            yield None
            return
        client = self.langfuse_client
        if client is None:
            yield None
            return

        observation_type = {
            "run": "agent",
            "turn": "generation",
            "tool": "tool",
        }.get(trace.name, "span")
        stack = ExitStack()
        try:
            observation = stack.enter_context(
                client.start_as_current_observation(
                    name=f"agent-native.{trace.name}",
                    as_type=observation_type,
                    metadata=self._redact_attrs(trace.attributes),
                )
            )
        except Exception:
            stack.close()
            yield None
            return
        try:
            if trace.name == "run":
                trace_id = client.get_current_trace_id()
                if trace_id:
                    self.langfuse_trace_ids[trace.run_id] = trace_id
            yield observation
        finally:
            stack.close()

    def _update_langfuse(self, observation: Any | None, trace: Trace) -> None:
        if observation is None:
            return
        attributes = self._redact_attrs(trace.attributes)
        update: dict[str, Any] = {"metadata": attributes}
        if trace.name == "run":
            update["output"] = {
                "status": attributes.get("status"),
                "turns": attributes.get("turns"),
            }
        elif trace.name == "turn":
            update["model"] = attributes.get("model")
            update["usage_details"] = {
                key: value
                for key, value in {
                    "input": attributes.get("input_tokens"),
                    "output": attributes.get("output_tokens"),
                    "total": attributes.get("total_tokens"),
                }.items()
                if value is not None
            }
            if attributes.get("cost") is not None:
                update["cost_details"] = {"total": attributes["cost"]}
        elif trace.name == "tool":
            update["output"] = attributes.get("output")
            if attributes.get("error"):
                update["level"] = "ERROR"
                update["status_message"] = str(attributes["error"])
        try:
            observation.update(**update)
        except Exception:
            return

    @contextmanager
    def _span(self, name: str, **attributes) -> Iterator[Trace]:
        parent_id = _CURRENT_SPAN.get("")
        trace = Trace(
            name=name,
            attributes=dict(attributes),
            run_id=_CURRENT_RUN.get(""),
            parent_id=parent_id,
        )
        if self.enabled:
            self.spans.append(trace)
            # A long-lived process would otherwise grow without bound. Dropping the
            # oldest spans loses history, which is the cheaper thing to lose.
            if len(self.spans) > self.max_spans:
                del self.spans[: len(self.spans) - self.max_spans]
        # Mark this span as the current parent for anything opened inside it. Set
        # even when disabled: the id costs nothing and keeps the two paths identical.
        token = _CURRENT_SPAN.set(trace.span_id)
        try:
            if not self.enabled:
                yield trace
            else:
                with self._langfuse_observation(trace) as observation:
                    yield trace
                    self._update_langfuse(observation, trace)
        finally:
            trace.finish()
            _CURRENT_SPAN.reset(token)

    @contextmanager
    def run_span(self, run_id: str, **attributes) -> Iterator[Trace]:
        """The outermost span. Also marks which run everything nested belongs to."""
        token = _CURRENT_RUN.set(run_id)
        try:
            with self._span("run", run_id=run_id, **attributes) as trace:
                yield trace
        finally:
            _CURRENT_RUN.reset(token)

    def turn_span(self, turn: int, **attributes) -> AbstractContextManager[Trace]:
        return self._span("turn", turn=turn, **attributes)

    def tool_span(self, tool: str, **attributes) -> AbstractContextManager[Trace]:
        return self._span("tool", tool=tool, **attributes)

    def shutdown(self) -> list:
        """Ship the recorded spans and let go. Returns the JSON paths written.

        Two independent sinks. If a collector is configured, the spans are replayed
        into OpenTelemetry and pushed to it; a collector that's down, or the
        `tracing` extra not being installed, is recorded on `otlp_skipped_reason`
        and costs the trace, not the run. If a `trace_dir` is set, the same spans
        are written as one JSON file per run. The return value is the JSON paths,
        so a CLI can tell the user where to look; the OTLP outcome is on the
        instance for anyone who wants it.
        """
        self.written = []
        if self.langfuse_client is not None:
            try:
                self.langfuse_client.flush()
            except Exception:
                pass
        self.otlp_attempted = False
        self.otlp_exported = False
        self.otlp_skipped_reason = ""

        spans, self.spans = self.spans, []
        if not spans:
            return self.written

        # Primary sink: a collector, if one is named. Never raises.
        if self._should_export_otlp():
            self.otlp_attempted = True
            self.otlp_exported, self.otlp_skipped_reason = self._export_otlp(spans)

        # Fallback sink: one JSON file per run on disk.
        if self.trace_dir is not None:
            self._write_json(spans)
        return self.written

    # -- the JSON fallback --------------------------------------------------
    def _redact_attrs(self, attributes: dict) -> dict:
        """Span attributes with any secret masked, or the attributes unchanged if
        no redactor is installed. Applied at export, so live spans stay exact."""
        if self.redactor is None:
            return attributes
        return self.redactor.redact(attributes)

    def _write_json(self, spans: list) -> None:
        """Write one JSON file per run under `trace_dir`. Drops quietly on OSError."""
        by_run: dict = {}
        for trace in spans:
            by_run.setdefault(trace.run_id, []).append(trace)

        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        except OSError:
            return

        for run_id, traces in by_run.items():
            # `start` is a monotonic clock reading, meaningless on its own, so
            # report each span as an offset from the first one in the file.
            origin = min(trace.start for trace in traces)
            payload = {
                "run_id": run_id,
                "written_at": datetime.now(UTC).isoformat(),
                "span_count": len(traces),
                "spans": [
                    {
                        "name": trace.name,
                        "started_after_s": round(trace.start - origin, 4),
                        "duration_s": round(trace.duration_s, 4),
                        "attributes": self._redact_attrs(trace.attributes),
                    }
                    for trace in traces
                ],
            }
            trace_dir = self.trace_dir
            if trace_dir is None:
                return
            path = trace_dir / f"{run_id or 'no-run'}.json"
            try:
                path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            except OSError:
                continue
            self.written.append(path)

    # -- the OpenTelemetry sink --------------------------------------------
    def _should_export_otlp(self) -> bool:
        """Is there anywhere to send spans? An explicit endpoint or the OTEL env vars."""
        return bool(
            self.otlp_endpoint
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        )

    def _export_otlp(self, spans: list) -> tuple[bool, str]:
        """Replay the recorded spans as OTel spans and push them to the collector.

        Returns `(exported, reason)`. `exported` is True only if the spans were
        handed to the exporter and flushed; `reason` explains a False - either the
        `tracing` extra isn't installed, or the export itself failed (a collector
        that's unreachable, say). Never raises: losing a trace must not fail a run,
        exactly as with the JSON path.

        Every span in one run shares a trace: the run's span is created as a root,
        and each turn and tool is created inside its parent's context, so the SDK
        threads one trace id through the lot. Timestamps are the real wall-clock
        pair (`start_wall` .. `start_wall + duration`), so the collector shows when
        the work happened, not just how long it took.
        """
        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError:
            return False, (
                "the 'tracing' extra is not installed "
                "(uv sync --all-packages --extra tracing); spans went to JSON only"
            )

        try:
            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)
            # An explicit endpoint is passed through; None lets the exporter read
            # the standard OTEL_EXPORTER_OTLP_* environment variables itself.
            exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint or None,
                headers=self.otlp_headers or None,
            )
            # Simple, not batched: this is a one-shot export at shutdown, so
            # exporting on span end and then flushing is the whole lifecycle.
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            tracer = provider.get_tracer(self.service_name)

            # Create parent-before-child. `_span` appends a parent to the list
            # before any child runs inside it, so creation order already satisfies
            # this; a child whose parent is somehow missing is created as a root
            # rather than dropped.
            made: dict[str, Any] = {}
            for tr in spans:
                parent = made.get(tr.parent_id)
                ctx = ot_trace.set_span_in_context(parent) if parent is not None else None
                span = tracer.start_span(
                    tr.name,
                    context=ctx,
                    start_time=int(tr.start_wall * 1_000_000_000),
                    attributes=_otlp_attributes(self._redact_attrs(tr.attributes)),
                )
                made[tr.span_id] = span

            # End children before parents, so a parent's duration still encloses
            # them in the exported trace.
            for tr in reversed(spans):
                exported_span = made.get(tr.span_id)
                if exported_span is not None:
                    exported_span.end(
                        end_time=int((tr.start_wall + tr.duration_s) * 1_000_000_000)
                    )

            provider.force_flush()
            provider.shutdown()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""


def _otlp_attributes(attributes: dict) -> dict:
    """Coerce span attributes to what OTel accepts (str/bool/int/float or those).

    Anything else - a dict of tool arguments, say - is stringified rather than
    dropped, so the collector still shows it even if it can't type it.
    """
    out: dict = {}
    for key, value in attributes.items():
        if isinstance(value, (str, bool, int, float)):
            out[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, bool, int, float)) for v in value
        ):
            out[key] = list(value)
        else:
            out[key] = str(value)
    return out

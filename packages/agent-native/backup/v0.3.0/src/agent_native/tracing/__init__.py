"""TracingService - OpenTelemetry GenAI spans exported to Langfuse.

No existing tracing helper exists anywhere in the repo (packages/observability
is an empty scaffold), so this is built here per the task instructions
("wrap, don't rewrite" does not apply). OTel deps are an optional extra
(`uv sync --extra tracing`); when software is missing or tracing is disabled,
every method is a no-op - tests and offline runs are unaffected.

Span naming follows the GenAI semantic conventions used by Langfuse:
gen_ai.operation.name = "chat" / "tool", gen_ai.request.model, etc.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings


@dataclass(slots=True)
class Span:
    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    _otel: Any = None


class TracingService:
    def __init__(self, settings: Settings | None = None, enabled: bool | None = None) -> None:
        self.enabled = enabled if enabled is not None else (settings.tracing_enabled if settings else False)
        self.tracer = None
        if self.enabled and settings is not None:
            self._init_otel(settings)

    def _init_otel(self, settings: Settings) -> None:
        try:
            from opentelemetry import trace  # type: ignore
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
            from opentelemetry.sdk.resources import Resource  # type: ignore
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

            resource = Resource.create({"service.name": "agent-native"})
            provider = TracerProvider(resource=resource)
            endpoint = settings.langfuse_url or "http://localhost:3000/api/public/otel"
            headers = {}
            if settings.langfuse_public_key and settings.langfuse_secret_key:
                headers = {
                    "Authorization": f"Basic {_basic_auth(settings.langfuse_public_key, settings.langfuse_secret_key)}",
                }
            provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, headers=headers)
            ))
            trace.set_tracer_provider(provider)  # type: ignore[attr-defined]
            self.tracer = trace.get_tracer("agent-native")  # type: ignore[attr-defined]
        except Exception:
            self.enabled = False
            self.tracer = None

    def start_span(self, name: str, inputs: dict[str, Any] | None = None) -> Span:
        span = Span(name=name, inputs=inputs or {})
        if self.enabled and self.tracer is not None:
            otel = self.tracer.start_span(name, attributes=_genai_attributes(name, inputs))
            span._otel = otel
        return span

    def end_span(self, span: Span, outputs: dict[str, Any] | None = None) -> None:
        if outputs:
            span.outputs = outputs
        if span._otel is not None:
            span._otel.end()

    @contextlib.contextmanager
    def span(self, name: str, inputs: dict[str, Any] | None = None):
        span = self.start_span(name, inputs)
        try:
            yield span
        finally:
            self.end_span(span)


def _genai_attributes(name: str, inputs: dict[str, Any] | None) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if name == "llm":
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.model"] = (inputs or {}).get("model", "groq")
    elif name == "tool":
        attrs["gen_ai.operation.name"] = "tool"
        attrs["gen_ai.tool.name"] = (inputs or {}).get("tool_name", "")
    return attrs


def _basic_auth(public_key: str, secret_key: str) -> str:
    import base64

    return base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

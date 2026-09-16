"""Data masking for traces.

Langfuse applies the client's ``mask`` callable to every observation input and
output before ingestion. This redacts common secret-bearing keys and obvious
credential patterns so they never leave the process, satisfying the
"sensitive data masked" tracing baseline.
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from typing import Any

# Substring match (case-insensitive) against dict keys whose values are secrets.
_SENSITIVE_KEY_HINTS: tuple[str, ...] = (
    "api_key", "apikey", "secret", "password", "passwd", "token",
    "authorization", "auth", "credential", "private_key", "access_key",
)

# Value patterns that look like credentials/PII regardless of their key.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),          # OpenAI-style secret keys
    re.compile(r"sk-lf-[A-Za-z0-9_\-]{8,}"),        # Langfuse secret keys
    re.compile(r"pk-lf-[A-Za-z0-9_\-]{8,}"),        # Langfuse public keys
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{10,}"),   # bearer tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),      # GitHub tokens
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email (PII)
)

_REDACTED = "[REDACTED]"


def _looks_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SENSITIVE_KEY_HINTS)


def _mask_str(value: str) -> str:
    masked = value
    for pattern in _SECRET_VALUE_PATTERNS:
        masked = pattern.sub(_REDACTED, masked)
    return masked


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        return _mask_str(value)
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _looks_sensitive(str(k)) else _mask_value(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_mask_value(v) for v in value)
    return value


def mask(*, data: Any, **kwargs: Any) -> Any:
    """Mask secrets in trace data. Signature matches Langfuse's ``mask`` hook.

    Never raises — masking failures must not break ingestion, so on any error
    the data is dropped to a safe placeholder rather than propagated raw.
    """
    try:
        return _mask_value(data)
    except Exception:
        return _REDACTED


def _mask_attribute(key: str, value: Any) -> Any:
    """Redact one OpenTelemetry span attribute.

    Mirrors ``_mask_value``'s policy at the flat attribute level: a
    sensitive-looking key redacts the whole value; otherwise string values (and
    strings inside a homogeneous sequence) are scrubbed for credential/PII
    patterns. Non-string scalars pass through.
    """
    if _looks_sensitive(key):
        return _REDACTED
    if isinstance(value, str):
        return _mask_str(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_mask_str(v) if isinstance(v, str) else v for v in value)
    return value


def mask_otel_spans(*, params: Any) -> Any:
    """Export-stage mask for raw OpenTelemetry span attributes.

    The ``mask`` hook only sees Langfuse observation inputs/outputs. Spans
    emitted by third-party OpenTelemetry instrumentations (HTTP clients, DB
    drivers, other GenAI SDKs) reach the exporter with their own attributes —
    e.g. ``http.request.header.authorization`` or a ``user.email`` — which
    ``mask`` never touches. This runs at export time and patches those.

    Returns sparse per-span patches (only changed attributes). Raising here
    makes Langfuse drop the whole batch, which is fail-closed, so this is left
    to propagate rather than swallow-and-export-unmasked.
    """
    # Importing ``langfuse.types`` eagerly makes *every* API startup import the
    # complete SDK. On Python 3.13 that import can spend a long time generating
    # dataclass methods, even when tracing is disabled. A real Langfuse callback
    # necessarily passes an SDK params object, so its types module is already
    # loaded by the time this hook is invoked. Unit callers receive compatible
    # lightweight objects without importing the optional SDK at all.
    types_module = sys.modules.get("langfuse.types")
    result_type = getattr(types_module, "MaskOtelSpansResult", None)
    patch_type = getattr(types_module, "OtelSpanPatch", None)
    patches: dict[Any, Any] = {}
    for identifier, span in params.spans.items():
        changed: dict[str, Any] = {}
        for key, value in span.attributes.items():
            masked = _mask_attribute(str(key), value)
            if masked != value:
                changed[key] = masked
        if changed:
            factory = patch_type or SimpleNamespace
            patches[identifier] = factory(set_attributes=changed)
    factory = result_type or SimpleNamespace
    return factory(span_patches=patches)

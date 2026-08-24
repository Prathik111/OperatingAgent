"""Data masking for traces.

Langfuse applies the client's ``mask`` callable to every observation input and
output before ingestion. This redacts common secret-bearing keys and obvious
credential patterns so they never leave the process, satisfying the
"sensitive data masked" tracing baseline.
"""

from __future__ import annotations

import re
from typing import Any
from langfuse.types import MaskFunction

# Substring match (case-insensitive) against dict keys whose values are secrets.
_SENSITIVE_KEY_HINTS: tuple[str, ...] = (
    "api_key", "apikey", "secret", "password", "passwd", "token",
    "authorization", "auth", "credential", "private_key", "access_key",
)

# Value patterns that look like credentials regardless of their key.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),          # OpenAI-style secret keys
    re.compile(r"sk-lf-[A-Za-z0-9_\-]{8,}"),        # Langfuse secret keys
    re.compile(r"pk-lf-[A-Za-z0-9_\-]{8,}"),        # Langfuse public keys
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{10,}"),   # bearer tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),      # GitHub tokens
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

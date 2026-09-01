"""Redaction: keep secrets out of everything that gets stored, shipped, or shown.

The only secret this project actually holds is the Groq API key, read from
`GROQ_API_KEY` (see models/groq_model.py). But a key doesn't leak by being held -
it leaks by being *written down somewhere it's read back*: an event row a UI
replays, a trace shipped to a collector, a JSON file on disk, a memory the agent
keeps. The realistic path is mundane - the agent runs `printenv`, the key is in
the tool's output, and now it's in an event and a trace.

So redaction happens at those *serializing* boundaries, not in the conversation
itself. The agent's own messages are left exact on purpose: they're the state it
reasons over, and masking them would corrupt it. Everything that leaves the
process - or is stored to be read back later - runs through a `Redactor` first.

Two layers, because each covers the other's gap:

* **Known values.** A `SecretSource` names the exact strings to mask - by default
  the value of `GROQ_API_KEY`, read from the environment. This catches the real
  leak (a key dumped whole in tool output) with certainty: we know the string to
  look for, so its shape doesn't matter.
* **Key-shaped patterns.** A backstop for keys we *don't* hold the value of - a
  second provider's key the model emitted, a bearer token in a header, a
  `SECRET=...` line in an env dump. Matched by shape, not by value.

Masking is longest-value-first, so one secret containing another can't leave a
tail behind, and it's idempotent: the mask is not itself key-shaped and equals no
secret, so redacting already-redacted text changes nothing. That last property is
what makes re-publishing a stored event on resume (step 11) safe.
"""

from __future__ import annotations

import os
import re
from typing import Any

#: What a redacted secret is replaced with. Deliberately not key-shaped and equal
#: to no real value, so a second pass over redacted text is a no-op.
MASK = "[redacted]"

#: Shortest value worth masking. Below this a "secret" is as likely a false match
#: (an empty or placeholder env var) as a real key, and masking it just mangles text.
_MIN_VALUE_LEN = 6


class SecretSource:
    """Where the exact secret strings to mask come from. Env is the default one."""

    def values(self) -> list[str]:  # pragma: no cover - overridden
        return []


class EnvSecretSource(SecretSource):
    """The secret values are whatever these environment variables hold, right now.

    Read at redaction time rather than cached, so a key loaded from a `.env` file
    after the runtime was built (the common case - `load_env()` runs lazily) is
    still masked. Empty or trivially short values are ignored, so an unset
    `GROQ_API_KEY` doesn't turn `""` into something we try to mask everywhere.
    """

    def __init__(self, var_names: tuple[str, ...] = ("GROQ_API_KEY",)) -> None:
        self._var_names = tuple(var_names)

    def values(self) -> list[str]:
        found = []
        for name in self._var_names:
            value = os.environ.get(name) or ""
            if len(value) >= _MIN_VALUE_LEN:
                found.append(value)
        return found


class StaticSecretSource(SecretSource):
    """A fixed set of secret strings. Handy for tests and for callers that hold a
    secret in hand rather than in the environment."""

    def __init__(self, values: list[str] | tuple[str, ...]) -> None:
        self._values = [v for v in values if v and len(v) >= _MIN_VALUE_LEN]

    def values(self) -> list[str]:
        return list(self._values)


# -- key-shaped patterns ----------------------------------------------------
# Groq keys start `gsk_`; OpenAI/Anthropic-style keys start `sk-` (with optional
# `proj-`/`ant-` infixes). Both are long runs of base62 after the prefix.
_GROQ = re.compile(r"gsk_[A-Za-z0-9]{16,}")
_SK = re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9]{16,}")
# `Authorization: Bearer <token>` - keep the word, drop the token.
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}")
# A secret-ish name assigned a value: `GROQ_API_KEY=gsk_...`, `password: hunter2`.
# Only names that look like secrets trip it, so `count=5` is left alone.
_ASSIGN = re.compile(
    r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|secret|token|password|passwd|pwd))(\s*[=:]\s*)(\S+)"
)


def _redact_patterns(text: str, mask: str) -> str:
    text = _GROQ.sub(mask, text)
    text = _SK.sub(mask, text)
    text = _BEARER.sub(lambda m: m.group(1) + mask, text)
    text = _ASSIGN.sub(lambda m: m.group(1) + m.group(2) + mask, text)
    return text


class Redactor:
    """Masks known secret values and key-shaped strings anywhere in a value.

    `redact` walks dicts, lists and tuples so a whole event `data` payload or a
    span's attribute map can be handed in as-is; strings are masked, everything
    else is returned unchanged. The input is never mutated - a new structure comes
    back - so the caller's live objects (a span still being inspected in-process)
    keep their exact values.
    """

    def __init__(self, source: SecretSource | None = None, mask: str = MASK) -> None:
        self._source = source if source is not None else EnvSecretSource()
        self._mask = mask

    def redact_text(self, text: Any) -> Any:
        if not isinstance(text, str) or not text:
            return text
        out = text
        # Known values first, longest first, so a secret that contains a shorter
        # one is masked whole rather than leaving its tail behind.
        for value in sorted(self._source.values(), key=len, reverse=True):
            if value and value in out:
                out = out.replace(value, self._mask)
        return _redact_patterns(out, self._mask)

    def redact(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {key: self.redact(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.redact(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact(value) for value in obj)
        return obj

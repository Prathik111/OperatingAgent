"""Tests for ``observability.masking``.

Masking is the last line of defence before trace data leaves the process, so
the tests cover secret-bearing keys, credential-shaped values, recursion into
nested structures, container-type preservation, and the never-raises contract.
"""

from __future__ import annotations

import pytest
from observability.masking import mask, mask_otel_spans

REDACTED = "[REDACTED]"


# ---------------------------------------------------------------------------
# Key-based redaction
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize(
    "key",
    [
        "api_key", "apikey", "API_KEY", "secret", "password", "passwd",
        "token", "authorization", "auth", "credential", "private_key",
        "access_key", "user_api_key", "X-Auth-Token",
    ],
)
def test_sensitive_keys_are_redacted_regardless_of_value(key: str) -> None:
    assert mask(data={key: "whatever"}) == {key: REDACTED}


def test_non_sensitive_keys_are_preserved() -> None:
    data = {"username": "alice", "count": 3, "enabled": True}
    assert mask(data=data) == data


# ---------------------------------------------------------------------------
# Value-pattern redaction
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize(
    "value",
    [
        "sk-ABCDEFGHIJKLMNOP1234",
        "sk-lf-abcdefgh",
        "pk-lf-abcdefgh",
        "Bearer abcdef1234567890",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    ],
)
def test_credential_shaped_values_are_redacted_by_pattern(value: str) -> None:
    """Even under a harmless key, a credential-shaped value is scrubbed."""
    masked = mask(data={"note": value})
    assert masked["note"] == REDACTED


@pytest.mark.regression
def test_secret_embedded_in_text_is_replaced_in_place() -> None:
    masked = mask(data="token is sk-ABCDEFGHIJKLMNOP1234 ok")
    assert "sk-ABCDEFGHIJKLMNOP1234" not in masked
    assert REDACTED in masked
    assert masked.startswith("token is ") and masked.endswith(" ok")


def test_ordinary_strings_pass_through() -> None:
    assert mask(data="just a normal message") == "just a normal message"


# ---------------------------------------------------------------------------
# Email / PII redaction
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_email_value_under_harmless_key_is_redacted() -> None:
    """An email is PII even under a non-sensitive key like ``contact``."""
    assert mask(data={"contact": "jane.doe@example.com"}) == {"contact": REDACTED}


@pytest.mark.regression
def test_email_embedded_in_free_text_is_replaced_in_place() -> None:
    masked = mask(data="ping me at jane.doe@example.com anytime")
    assert "jane.doe@example.com" not in masked
    assert REDACTED in masked
    assert masked.startswith("ping me at ") and masked.endswith(" anytime")


# ---------------------------------------------------------------------------
# Recursion & container handling
# ---------------------------------------------------------------------------


def test_nested_structures_are_masked_recursively() -> None:
    data = {
        "outer": {"password": "hunter2", "safe": "keep"},
        "list": [{"token": "x"}, "sk-ABCDEFGHIJKLMNOP1234"],
    }
    masked = mask(data=data)
    assert masked["outer"]["password"] == REDACTED
    assert masked["outer"]["safe"] == "keep"
    assert masked["list"][0]["token"] == REDACTED
    assert masked["list"][1] == REDACTED


def test_list_and_tuple_types_are_preserved() -> None:
    assert isinstance(mask(data=["a", "b"]), list)
    result = mask(data=("a", "b"))
    assert isinstance(result, tuple)


def test_non_string_scalars_pass_through_untouched() -> None:
    assert mask(data=42) == 42
    assert mask(data=3.14) == 3.14
    assert mask(data=True) is True
    assert mask(data=None) is None


def test_non_string_dict_keys_are_handled() -> None:
    """Keys are stringified before the sensitivity check, so a non-string key
    must not blow up the mask."""
    assert mask(data={1: "value", 2: "other"}) == {1: "value", 2: "other"}


# ---------------------------------------------------------------------------
# Never-raises contract
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_mask_never_raises_returns_placeholder_on_error() -> None:
    """If traversal explodes, masking must degrade to the safe placeholder
    rather than propagating — an ingestion break is worse than over-redaction."""

    class Boom:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    # A dict whose *value* recursion hits an object that raises when stringified
    # via a nested secret pattern path is hard to trigger; instead force the
    # top-level traversal to raise by making .items() blow up.
    class ExplodingDict(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("boom")

    assert mask(data=ExplodingDict(a=1)) == REDACTED


def test_mask_accepts_extra_kwargs() -> None:
    """Langfuse may pass extra keyword arguments; the signature tolerates them."""
    assert mask(data="plain", extra="ignored", another=1) == "plain"


# ---------------------------------------------------------------------------
# Export-stage OpenTelemetry span masking (mask_otel_spans)
# ---------------------------------------------------------------------------


class _FakeSpan:
    """Minimal stand-in for ``OtelSpanData`` — mask_otel_spans only reads
    ``.attributes``, so a real (many-field, frozen) snapshot isn't needed."""

    def __init__(self, attributes: dict) -> None:
        self.attributes = attributes


class _FakeParams:
    def __init__(self, spans: dict) -> None:
        self.spans = spans


@pytest.mark.regression
def test_mask_otel_spans_redacts_secret_and_pii_in_third_party_span() -> None:
    """A span from a third-party OTel instrumentation carrying both a bearer
    token and an email must have both scrubbed; benign attributes stay out of
    the sparse patch."""
    span = _FakeSpan(
        {
            "http.request.header.authorization": "Bearer abcdef1234567890",
            "user.email": "jane.doe@example.com",
            "gen_ai.system": "openai",
        }
    )

    result = mask_otel_spans(params=_FakeParams({"span-1": span}))

    patch = result.span_patches["span-1"]
    assert patch.set_attributes["http.request.header.authorization"] == REDACTED
    assert patch.set_attributes["user.email"] == REDACTED
    # Untouched attribute must not appear in the sparse patch.
    assert "gen_ai.system" not in patch.set_attributes


def test_mask_otel_spans_leaves_clean_span_unpatched() -> None:
    span = _FakeSpan({"gen_ai.system": "openai", "gen_ai.request.model": "gpt-x"})
    result = mask_otel_spans(params=_FakeParams({"s": span}))
    assert "s" not in result.span_patches

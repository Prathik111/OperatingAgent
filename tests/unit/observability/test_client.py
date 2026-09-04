"""Tests for ``observability.client``.

The client is the single Langfuse initialisation point and must satisfy three
guarantees the rest of the app leans on:

* **degrade, never crash** — missing credentials or an SDK that raises during
  construction leave the app running untraced;
* **idempotent** — ``init_tracing`` may be called from several entry points;
* **lazy** — ``get_client`` initialises from env on first use.

The Langfuse SDK is patched out so these tests never open a socket. The
module-level ``_client`` / ``_initialised`` globals are reset around every test
by the autouse ``reset_observability_singletons`` fixture in the root conftest.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from observability import client as client_module
from observability.masking import mask, mask_otel_spans
from observability.settings import LangfuseSettings


def enabled_settings() -> LangfuseSettings:
    return LangfuseSettings(
        public_key="pk-lf-x",
        secret_key="sk-lf-y",
        host="https://host",
        environment="test",
        release="rel-1",
    )


def disabled_settings() -> LangfuseSettings:
    return LangfuseSettings(
        public_key=None, secret_key=None,
        host="https://host", environment="test", release=None,
    )


class FakeLangfuse:
    """Records constructor kwargs and calls; stands in for ``langfuse.Langfuse``."""

    instances: ClassVar[list[FakeLangfuse]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.flushed = 0
        self.shut_down = 0
        FakeLangfuse.instances.append(self)

    def flush(self) -> None:
        self.flushed += 1

    def shutdown(self) -> None:
        self.shut_down += 1


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    FakeLangfuse.instances.clear()


@pytest.fixture
def patch_langfuse(monkeypatch: pytest.MonkeyPatch) -> type[FakeLangfuse]:
    """Replace ``langfuse.Langfuse`` so ``from langfuse import Langfuse`` binds
    the fake — no real client is ever constructed."""
    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse, raising=False)
    return FakeLangfuse


# ---------------------------------------------------------------------------
# init_tracing
# ---------------------------------------------------------------------------


def test_init_disabled_returns_none() -> None:
    assert client_module.init_tracing(disabled_settings()) is None
    assert client_module._initialised is True
    assert client_module._client is None


def test_init_enabled_constructs_client_with_expected_kwargs(
    patch_langfuse: type[FakeLangfuse],
) -> None:
    result = client_module.init_tracing(enabled_settings())
    assert isinstance(result, FakeLangfuse)
    assert result.kwargs == {
        "public_key": "pk-lf-x",
        "secret_key": "sk-lf-y",
        "host": "https://host",
        "environment": "test",
        "release": "rel-1",
        "mask": mask,
        "mask_otel_spans": mask_otel_spans,
    }


def test_init_is_idempotent(patch_langfuse: type[FakeLangfuse]) -> None:
    first = client_module.init_tracing(enabled_settings())
    second = client_module.init_tracing(enabled_settings())
    assert first is second
    assert len(FakeLangfuse.instances) == 1  # constructed once only


def test_init_swallows_sdk_construction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure inside the SDK constructor must not propagate; the app simply
    runs untraced."""
    import langfuse

    def boom(**_: Any) -> None:
        raise RuntimeError("cannot reach langfuse")

    monkeypatch.setattr(langfuse, "Langfuse", boom, raising=False)
    assert client_module.init_tracing(enabled_settings()) is None
    assert client_module._initialised is True


def test_init_reads_env_when_no_settings_passed(
    patch_langfuse: type[FakeLangfuse], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-env")
    client_module.init_tracing()
    assert FakeLangfuse.instances[0].kwargs["public_key"] == "pk-lf-env"


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


def test_get_client_lazily_initialises(patch_langfuse: type[FakeLangfuse]) -> None:
    assert client_module._initialised is False
    got = client_module.get_client()  # reads env; conftest cleared creds -> disabled
    assert got is None
    assert client_module._initialised is True


def test_get_client_returns_existing_after_init(patch_langfuse: type[FakeLangfuse]) -> None:
    created = client_module.init_tracing(enabled_settings())
    assert client_module.get_client() is created


# ---------------------------------------------------------------------------
# get_callback_handler
# ---------------------------------------------------------------------------


def test_callback_handler_none_when_disabled() -> None:
    client_module.init_tracing(disabled_settings())
    assert client_module.get_callback_handler() is None


def test_callback_handler_returned_when_enabled(
    patch_langfuse: type[FakeLangfuse], fake_module: Any
) -> None:
    class FakeHandler:
        pass

    fake_module("langfuse.langchain", CallbackHandler=FakeHandler)
    client_module.init_tracing(enabled_settings())
    handler = client_module.get_callback_handler()
    assert isinstance(handler, FakeHandler)


def test_callback_handler_none_when_creation_fails(
    patch_langfuse: type[FakeLangfuse], fake_module: Any
) -> None:
    def boom() -> None:
        raise RuntimeError("no handler")

    fake_module("langfuse.langchain", CallbackHandler=boom)
    client_module.init_tracing(enabled_settings())
    assert client_module.get_callback_handler() is None


# ---------------------------------------------------------------------------
# flush / shutdown
# ---------------------------------------------------------------------------


def test_flush_and_shutdown_delegate_to_client(patch_langfuse: type[FakeLangfuse]) -> None:
    created = client_module.init_tracing(enabled_settings())
    client_module.flush()
    client_module.shutdown()
    assert created.flushed == 1
    assert created.shut_down == 1


def test_flush_and_shutdown_are_noops_when_disabled() -> None:
    client_module.init_tracing(disabled_settings())
    # Must not raise despite there being no client.
    client_module.flush()
    client_module.shutdown()


def test_flush_swallows_client_error(patch_langfuse: type[FakeLangfuse]) -> None:
    created = client_module.init_tracing(enabled_settings())

    def boom() -> None:
        raise RuntimeError("flush failed")

    created.flush = boom  # type: ignore[method-assign]
    client_module.flush()  # must not raise

from __future__ import annotations

import asyncio

import api


def test_windows_server_uses_selector_loop(monkeypatch) -> None:
    monkeypatch.setattr(api.sys, "platform", "win32")

    loop_factory = api._server_loop()

    assert loop_factory is asyncio.SelectorEventLoop


def test_custom_selector_factory_matches_uvicorn_contract() -> None:
    loop = api._selector_loop_factory()
    assert isinstance(loop, asyncio.SelectorEventLoop)
    loop.close()


def test_non_windows_server_keeps_uvicorn_auto_loop(monkeypatch) -> None:
    monkeypatch.setattr(api.sys, "platform", "linux")

    assert api._server_loop() == "auto"

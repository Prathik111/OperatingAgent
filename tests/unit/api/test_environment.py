from __future__ import annotations

import sys

import api
from api import create_app
from api.environment import load_environment


def test_load_environment_reads_nearest_dotenv(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("API_PORT=9123\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("API_PORT", raising=False)

    load_environment()

    assert create_app().state.settings.port == 9123


def test_load_environment_preserves_exported_values(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("API_PORT=9123\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_PORT", "9456")

    load_environment()

    assert create_app().state.settings.port == 9456


def test_api_entrypoint_loads_bind_settings_from_dotenv(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / ".env").write_text(
        "API_HOST=0.0.0.0\nAPI_PORT=9234\nAPI_LOG_LEVEL=warning\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for name in ("API_HOST", "API_PORT", "API_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)

    received: dict = {}

    def fake_run(application: str, **kwargs) -> None:
        received["application"] = application
        received.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    api.main()

    assert received == {
        "application": "api.app:create_app",
        "factory": True,
        "host": "0.0.0.0",
        "port": 9234,
        "log_level": "warning",
        "loop": "api:_selector_loop_factory" if sys.platform == "win32" else "auto",
    }

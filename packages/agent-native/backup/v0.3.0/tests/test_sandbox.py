"""SandboxManager tests - config/egress logic always; Docker lifecycle gated
on docker availability (decision #6: deny-all egress + allowlist config)."""

from __future__ import annotations

import pytest

from agent_native.config import SandboxConfig, is_host_allowed
from agent_native.sandbox import SandboxManager, SandboxSession


def test_is_host_allowed_deny_defaults_to_false():
    cfg = SandboxConfig(egress="deny", allowed_hosts=["pypi.org"])
    assert is_host_allowed("pypi.org", cfg)
    assert is_host_allowed("files.pypi.org", cfg)
    assert not is_host_allowed("evil.example.com", cfg)


def test_is_host_allowed_allow_egress_everything():
    cfg = SandboxConfig(egress="allow", allowed_hosts=[])
    assert is_host_allowed("anything.example", cfg)


def test_sandbox_config_env_allowlist_extends_file(tmp_path, monkeypatch):
    import agent_native.config as cfg_mod

    conf = tmp_path / "c.toml"
    conf.write_text(
        '[sandbox]\negress = "deny"\nallowed_hosts = ["pypi.org"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_NATIVE_CONFIG", str(conf))
    monkeypatch.setenv("AGENT_NATIVE_SANDBOX_ALLOWED_HOSTS", "npmjs.org, docs.python.org")
    settings = cfg_mod.load_settings()
    assert "pypi.org" in settings.sandbox.allowed_hosts
    assert "npmjs.org" in settings.sandbox.allowed_hosts
    assert is_host_allowed("docs.python.org", settings.sandbox)
    assert not is_host_allowed("other.example", settings.sandbox)


def test_sandbox_config_env_egress_override(tmp_path, monkeypatch):
    import agent_native.config as cfg_mod

    conf = tmp_path / "c.toml"
    conf.write_text('[sandbox]\negress = "deny"\n', encoding="utf-8")
    monkeypatch.setenv("AGENT_NATIVE_CONFIG", str(conf))
    monkeypatch.setenv("AGENT_NATIVE_SANDBOX_EGRESS", "allow")
    settings = cfg_mod.load_settings()
    assert settings.sandbox.egress == "allow"


def test_time_and_cpu_limits_from_config(tmp_path, monkeypatch):
    import agent_native.config as cfg_mod

    conf = tmp_path / "c.toml"
    conf.write_text(
        '[sandbox]\ncpu_limit = "0.5"\ntime_limit_s = 15\nmemory_limit = "256m"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_NATIVE_CONFIG", str(conf))
    settings = cfg_mod.load_settings()
    assert settings.sandbox.cpu_limit == "0.5"
    assert settings.sandbox.time_limit_s == 15
    assert settings.sandbox.memory_limit == "256m"


@pytest.mark.skipif(not SandboxManager.docker_available(), reason="docker not available")
def test_docker_lifecycle(tmp_path):
    manager = SandboxManager(SandboxConfig(enabled=True))
    session = manager.create_session("docker-test", workspace=tmp_path)
    assert isinstance(session, SandboxSession)
    assert session.workspace_path == tmp_path
    verdict = manager.run_command("docker-test", ["echo", "hello"], timeout_s=20)
    assert verdict.returncode == 0
    assert "hello" in verdict.stdout
    manager.destroy_session("docker-test")
    assert manager.get_session("docker-test") is None


def test_create_session_without_docker_raises(tmp_path, monkeypatch):
    manager = SandboxManager(SandboxConfig(enabled=True))

    def fake_available() -> bool:
        return False

    monkeypatch.setattr(SandboxManager, "docker_available", staticmethod(fake_available))
    from agent_native.sandbox import SandboxError

    with pytest.raises(SandboxError):
        manager.create_session("t", tmp_path)


def test_run_command_without_session_raises():
    manager = SandboxManager(SandboxConfig(enabled=True))
    from agent_native.sandbox import SandboxError

    with pytest.raises(SandboxError):
        manager.run_command("nope", ["ls"])
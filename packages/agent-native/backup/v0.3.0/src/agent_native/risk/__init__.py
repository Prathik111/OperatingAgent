"""Deterministic, rule-based risk classification of tool calls.

Decision #5 (per user): the classifier is NOT stateless - it keeps per-task
session history and can escalate a call whose *sequence* is risky even when
each individual call would be SAFE/REVIEW on its own.

Explicitly *not* LLM-judged: rules are regex/allowlist on (tool name, args).
The guarantee that matters is BLOCKED is a hard veto ReactExecutor honors.

Session rules (documented, tested):
  R1  exfil shape:  a secret/credential read earlier in the session followed
      by a network-send call -> BLOCKED (even though send alone is REVIEW).
  R2  destroy-then-publish: a destructive delete earlier in the session
      followed by a network-send call -> BLOCKED.
  R3  repeat offender: an identical call (same tool name AND same arguments)
      already BLOCKED this session is BLOCKED again - the model keeps trying
      the thing that was vetoed, so we keep vetoing it.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from typing import Callable

from ..types import RiskLevel, ToolCallRequest

_BLOCKED_UNCONDITIONAL: frozenset[str] = frozenset({
    "shutdown", "reboot", "poweroff", "mkfs", "format", "fdisk",
    "drop_database", "drop_table",
})

_DESTRUCTIVE_DELETE: frozenset[str] = frozenset({
    "delete_file", "delete_directory", "remove", "rmdir", "rm",
})

_NETWORK_SEND: frozenset[str] = frozenset({
    "git_push", "push", "scp", "sftp", "upload_file",
})

_FORCE_PUSH_TOOLS: frozenset[str] = frozenset({
    "git_force_push", "force_push",
})

_SECRET_PATH_RE = re.compile(
    r"\.env(\.|$)|id_rsa|id_ed25519|\.pem$|\.key$|credentials|"
    r"\.aws/|\.ssh/|/etc/shadow|/etc/passwd$|secrets?/",
    re.IGNORECASE,
)

_FS_ROOT_RE = re.compile(r"(^|[=\s;|&])rm\s+-rf?\s+[/\\]\s*($|[=\s;|&])")
_WORKSPACE_ESCAPE_RE = re.compile(r"\.\.(/|\\)")
_SHELL_CHAIN_RE = re.compile(r"[;&|]{1,2}")
_CURL_POST_RE = re.compile(
    r"curl\s+.*(-X\s+(POST|PUT|PATCH)|--data|-d\s|--upload|-T\s|--post-file)",
    re.IGNORECASE,
)

_NETWORK_TOOLS: frozenset[str] = frozenset({
    "curl", "wget", "netcat", "nc", "pip_install", "npm_install", "git_clone",
})

_ENV_TOOLS: frozenset[str] = frozenset({
    "printenv", "env", "getenv", "read_env",
})

_ANALYSIS_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_directory", "exists", "metadata", "search_files",
    "grep", "git_status", "git_diff", "git_log", "ls", "cat", "find", "head", "tail",
    "query_knowledge", "retrieve_memory", "store_memory", "search_docs", "search_codebase",
})

_APPROVAL_REQUIRED_TOOLS_REVIEW: frozenset[str] = frozenset({
    "delete_file", "delete_directory", "remove", "rmdir", "move_file", "rename_file",
    "write_file", "edit_file", "git_commit_amend", "git_reset_hard", "git_checkout_discard",
    "git_branch_delete", "git_force_push", "sh", "bash", "pwsh", "powershell",
    "pip_install", "npm_install", "git_clone", "git_push", "chmod", "chown",
})

_SECRET_READ_TOOLS: frozenset[str] = frozenset({"read_file", "cat"})


class RiskClassifier:
    """Rule engine + per-task session history."""

    def __init__(
        self,
        allowlist_net_hosts: Callable[[str], bool] | None = None,
    ) -> None:
        """`allowlist_net_hosts(host) -> bool` gates hosts found in arguments.

        None disables host gating (pure rule tests); default wiring in
        NativeAgent uses config.is_host_allowed (deny-all egress unless the
        host is allowlisted - decision #6).
        """
        self._sessions: dict[str, deque[tuple[str, dict, RiskLevel]]] = {}
        self._lock = threading.Lock()
        self._allowlist_net_hosts = allowlist_net_hosts

    def start_session(self, task_id: str) -> None:
        with self._lock:
            self._sessions[task_id] = deque(maxlen=100)

    def end_session(self, task_id: str) -> None:
        with self._lock:
            self._sessions.pop(task_id, None)

    def session_history(self, task_id: str) -> list[tuple[str, dict, RiskLevel]]:
        with self._lock:
            return list(self._sessions.get(task_id, ()))

    def classify(self, task_id: str, request: ToolCallRequest) -> RiskLevel:
        if self._allowlist_net_hosts is not None and not self._host_ok(request):
            level = RiskLevel.BLOCKED
        else:
            level = self._classify_single(task_id, request)
        with self._lock:
            history = self._sessions.setdefault(task_id, deque(maxlen=100))
            history.append((request.tool_name, request.arguments, level))
        return level

    # ---- internals -------------------------------------------------------

    def _host_ok(self, request: ToolCallRequest) -> bool:
        host = _extract_host(request)
        if host is None:
            return True
        return bool(self._allowlist_net_hosts and self._allowlist_net_hosts(host))

    def _classify_single(self, task_id: str, request: ToolCallRequest) -> RiskLevel:
        name, args = request.tool_name, request.arguments
        combined = f"{name} {_args_text(args)}"

        if name in _BLOCKED_UNCONDITIONAL:
            return RiskLevel.BLOCKED
        if _FS_ROOT_RE.search(combined):
            return RiskLevel.BLOCKED

        is_network_send = self._is_network_send(name, args)
        history = self.session_history(task_id)

        if self._session_has_identical_blocked(history, (name, _args_text(args))):
            return RiskLevel.BLOCKED  # R3 repeat offender: identical call

        if self._session_read_secret(history) and is_network_send:
            return RiskLevel.BLOCKED  # R1 exfil shape

        if self._session_did_destructive_delete(history) and is_network_send:
            return RiskLevel.BLOCKED  # R2 destroy-then-publish

        if name in _NETWORK_SEND or _CURL_POST_RE.search(combined):
            return RiskLevel.REVIEW
        if name in _FORCE_PUSH_TOOLS:
            return RiskLevel.REVIEW
        if name in _NETWORK_TOOLS:
            return RiskLevel.REVIEW
        if name in _ENV_TOOLS:
            return RiskLevel.REVIEW
        if _SECRET_PATH_RE.search(_args_text(args)):
            return RiskLevel.REVIEW  # secret reads are REVIEW even for "safe" readers
        if name in _ANALYSIS_TOOLS:
            return RiskLevel.SAFE
        if name in _APPROVAL_REQUIRED_TOOLS_REVIEW:
            return RiskLevel.REVIEW
        if _WORKSPACE_ESCAPE_RE.search(_args_text(args)):
            return RiskLevel.REVIEW
        if name == "run_command" and _SHELL_CHAIN_RE.search(combined):
            return RiskLevel.REVIEW

        return RiskLevel.SAFE

    @staticmethod
    def _is_network_send(name: str, args: dict) -> bool:
        if name in _NETWORK_SEND:
            return True
        raw = _args_text(args)
        return bool(
            _CURL_POST_RE.search(raw)
            or (name in {"curl", "wget"} and any(flag in raw for flag in ("--post-file", "--upload", "--data", "-T ")))
        )

    def _session_read_secret(self, history) -> bool:
        for tool_name, args, _level in history:
            if tool_name in _ENV_TOOLS:
                return True
            if tool_name in _SECRET_READ_TOOLS and _SECRET_PATH_RE.search(_args_text(args)):
                return True
        return False

    @staticmethod
    def _session_did_destructive_delete(history) -> bool:
        return any(tool in _DESTRUCTIVE_DELETE for tool, _args, _level in history)

    @staticmethod
    def _session_has(history, level: RiskLevel, *names: str) -> bool:
        nameset = set(names)
        return any(lvl == level and (not nameset or tool in nameset) for tool, _a, lvl in history)

    @staticmethod
    def _session_has_identical_blocked(history, call: tuple[str, str]) -> bool:
        return any(lvl == RiskLevel.BLOCKED and (tool, _args_text(args)) == call
                   for tool, args, lvl in history)


def _extract_host(request: ToolCallRequest) -> str | None:
    """Best-effort host extraction for egress gating (None = no host to gate)."""
    raw = _args_text(request.arguments)
    m = re.search(r"https?://([^/:\s]+)|([\w.-]+):\d{2,5}([/\s]|$)", raw)
    if not m:
        return None
    return (m.group(1) or m.group(2)).strip().lower()


def _args_text(args: dict) -> str:
    if isinstance(args, dict):
        return " ".join(f"{k}={v}" for k, v in args.items())
    return str(args)

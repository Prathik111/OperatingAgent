"""Tools: argument checking, the dispatcher gate, and the built-ins."""

from __future__ import annotations

import os
import tempfile

from agent_native.config import AgentConfig
from agent_native.conversation import Session, ToolCall
from agent_native.database import MemoryDatabase
from agent_native.events import EventBus
from agent_native.loop import Cancellation, Limits, RunContext
from agent_native.permissions import (
    PermissionManager,
    PermissionStore,
    PolicyChain,
    RulePolicy,
    SessionPolicy,
)
from agent_native.tools.base import ArgumentChecker, ToolRegistry
from agent_native.tools.manager import ToolManager
from tests._fake_tools import default_tools


def _context(workdir: str) -> RunContext:
    session = Session(agent="build", working_directory=workdir)
    return RunContext(
        session=session, run_id="r1", config=AgentConfig(),
        limits=Limits(), cancellation=Cancellation(),
    )


def _manager():
    db = MemoryDatabase()
    bus = EventBus(db)
    registry = ToolRegistry()
    for tool in default_tools():
        registry.register(tool)
    permissions = PermissionManager(PermissionStore(db), bus)
    policy = PolicyChain([RulePolicy(), SessionPolicy()])
    return ToolManager(registry, policy, permissions), permissions


# -- ArgumentChecker ---------------------------------------------------------
async def test_argcheck_missing_required():
    ok, reason = ArgumentChecker().validate(
        {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}, {}
    )
    assert not ok and "path" in reason


async def test_argcheck_wrong_type():
    ok, _ = ArgumentChecker().validate(
        {"type": "object", "properties": {"n": {"type": "integer"}}}, {"n": "not-a-number"}
    )
    assert not ok


async def test_argcheck_ok():
    ok, reason = ArgumentChecker().validate(
        {"type": "object", "properties": {"p": {"type": "string"}}, "required": ["p"]}, {"p": "a"}
    )
    assert ok and reason == ""


# -- the dispatcher: every failure is a ToolResult, never a crash ------------
async def test_unknown_tool_is_a_result():
    manager, _ = _manager()
    result = await manager.execute(ToolCall(id="c", name="nope"), _context("."))
    assert not result.success and "Unknown tool" in result.error


async def test_bad_arguments_is_a_result():
    manager, _ = _manager()
    result = await manager.execute(ToolCall(id="c", name="read_file", arguments={}), _context("."))
    assert not result.success and "Invalid arguments" in result.error


async def test_read_file_allowed_without_asking():
    workdir = tempfile.mkdtemp()
    with open(os.path.join(workdir, "a.txt"), "w") as fh:
        fh.write("data")
    manager, _ = _manager()
    result = await manager.execute(
        ToolCall(id="c", name="read_file", arguments={"path": "a.txt"}), _context(workdir)
    )
    assert result.success and result.output == "data"


async def test_path_escape_is_blocked():
    workdir = tempfile.mkdtemp()
    manager, _ = _manager()
    result = await manager.execute(
        ToolCall(id="c", name="read_file", arguments={"path": "../../etc/passwd"}), _context(workdir)
    )
    assert not result.success and "outside" in result.error


async def test_write_asks_then_writes():
    import asyncio

    workdir = tempfile.mkdtemp()
    manager, permissions = _manager()
    call = ToolCall(id="c", name="write_file", arguments={"path": "o.txt", "content": "hi"})

    async def approve() -> None:
        while True:
            if permissions.pending():
                await permissions.resolve("c", True)
                return
            await asyncio.sleep(0.005)

    task = asyncio.create_task(approve())
    result = await manager.execute(call, _context(workdir))
    await task

    assert result.success
    with open(os.path.join(workdir, "o.txt")) as fh:
        assert fh.read() == "hi"

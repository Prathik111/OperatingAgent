"""ApprovalGateway tests - decision #4 (timeout -> auto-deny + event)."""

from __future__ import annotations

import asyncio

import pytest

from agent_native.approval import ApprovalGateway
from agent_native.events import APPROVAL_REQUESTED, APPROVAL_RESOLVED, APPROVAL_TIMED_OUT
from agent_native.types import ApprovalDecision, PlanStep, StepKind
from conftest import EventSink


def make_step() -> PlanStep:
    return PlanStep(id="s9", description="write file", kind=StepKind.TOOL, tool_name="write_file")


@pytest.mark.asyncio
async def test_approved_branch(sink: EventSink):
    gateway = ApprovalGateway(timeout_s=30, on_event=sink)
    step_id = make_step().id

    async def resolver():
        await asyncio.sleep(0.01)
        gateway.resolve(step_id, ApprovalDecision.APPROVED)

    t = asyncio.create_task(resolver())
    decision = await gateway.request_approval("t1", make_step())
    await t
    assert decision == ApprovalDecision.APPROVED
    assert APPROVAL_REQUESTED in sink.kinds()
    assert APPROVAL_RESOLVED in sink.kinds()
    assert APPROVAL_TIMED_OUT not in sink.kinds()
    assert gateway.pending == {}


@pytest.mark.asyncio
async def test_denied_branch(sink: EventSink):
    gateway = ApprovalGateway(timeout_s=30, on_event=sink)
    step = make_step()
    t = asyncio.create_task(gateway.request_approval("t1", step))
    await asyncio.sleep(0.01)  # let request_approval register the pending approval
    assert gateway.resolve(step.id, ApprovalDecision.DENIED)
    decision = await t
    assert decision == ApprovalDecision.DENIED
    assert APPROVAL_RESOLVED in sink.kinds()


@pytest.mark.asyncio
async def test_timeout_auto_denies_and_emits_event(sink: EventSink):
    gateway = ApprovalGateway(timeout_s=0.05, on_event=sink)
    decision = await gateway.request_approval("t1", make_step())
    assert decision == ApprovalDecision.TIMED_OUT
    kinds = sink.kinds()
    assert APPROVAL_REQUESTED in kinds
    assert APPROVAL_TIMED_OUT in kinds
    assert APPROVAL_RESOLVED not in kinds
    assert gateway.pending == {}


@pytest.mark.asyncio
async def test_resolve_unknown_step_is_noop():
    gateway = ApprovalGateway(timeout_s=30)
    assert gateway.resolve("nope", ApprovalDecision.APPROVED) is False


@pytest.mark.asyncio
async def test_close_flushes_pending_as_timed_out(sink: EventSink):
    gateway = ApprovalGateway(timeout_s=60, on_event=sink)

    async def wait_task():
        return await gateway.request_approval("t1", make_step())

    t = asyncio.create_task(wait_task())
    await asyncio.sleep(0.01)  # let it register before close() flushes
    await gateway.close()
    assert await t == ApprovalDecision.TIMED_OUT
    assert gateway.pending == {}
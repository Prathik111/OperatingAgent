from __future__ import annotations

from datetime import UTC, datetime

from common.events import AgentEvent, LLMCallRecord, ToolCallRecord


def test_tool_call_record_ignores_correlation_keys() -> None:
    # LangGraph tool_finished payloads carry call_id/step_id; native payloads
    # carry call_id/truncated. The service normalizes tool naming before this
    # point; from_payload itself must tolerate the correlation keys without
    # TypeError, and the event stream (not the record) stays the correlation
    # source.
    record = ToolCallRecord.from_payload(
        {
            "call_id": "call-1",
            "step_id": 42,
            "tool_name": "read_file",
            "arguments": {"path": "a.txt"},
            "success": True,
            "output": "ok",
            "truncated": False,
            "native_sequence": 7,
        }
    )
    assert record.tool_name == "read_file"
    assert record.success is True
    assert record.output == "ok"
    assert record.arguments == {"path": "a.txt"}


def test_tool_call_record_ignores_extra_keys_without_arguments() -> None:
    # A completed LangGraph tool event carries no arguments at all; the
    # service fills an empty dict, and the record must still build when the
    # key is absent entirely.
    record = ToolCallRecord.from_payload(
        {
            "call_id": "call-2",
            "tool_name": "write_file",
            "success": False,
            "error": "denied",
            "output": "",
        }
    )
    assert record.tool_name == "write_file"
    assert record.error == "denied"
    assert record.arguments is None


def test_llm_call_record_ignores_extra_keys() -> None:
    record = LLMCallRecord.from_payload(
        {
            "node_name": "responder",
            "provider": "groq",
            "model": "llama",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost": 0.01,
            "turn": 3,
            "native_run_id": "r",
            "started_at": datetime.now(UTC).isoformat(),
        }
    )
    assert record.prompt_tokens == 10
    assert isinstance(record.started_at, datetime)


def test_record_roundtrip_still_exact() -> None:
    record = ToolCallRecord(
        tool_name="git_status",
        arguments={"path": "."},
        success=True,
        risk_level="safe",
        started_at=datetime.now(UTC),
    )
    restored = ToolCallRecord.from_payload(record.to_payload())
    assert restored == record


def test_agent_event_payload_passthrough() -> None:
    event = AgentEvent(type="state", payload={"status": "running"})
    assert event.payload == {"status": "running"}

from __future__ import annotations

import os
import uuid

import pytest
from agent_native.conversation import Role, Session, media_part, user_message
from agent_native.events import Event
from agent_native.loop import RunRecord
from agent_native.memory import Memory
from agent_native.permissions import PermissionDuration, PermissionGrant
from agent_native.postgres import PostgresDatabase

DSN = os.getenv("OPERATING_AGENT_POSTGRES_DSN", os.getenv("DATABASE_URL", ""))
pytestmark = pytest.mark.skipif(not DSN, reason="set OPERATING_AGENT_POSTGRES_DSN for live Postgres")


async def test_canonical_postgres_adapter_retains_native_features() -> None:
    db = await PostgresDatabase.open(DSN)
    session = Session(
        id=f"integration-{uuid.uuid4()}",
        agent="build",
        title="canonical adapter",
        working_directory="C:/workspace/project",
        revision=3,
    )
    run_id = "run_" + uuid.uuid4().hex[:8]
    memory = Memory(text="remember this", session_id=session.id)
    try:
        await db.create_session(session)
        restored = await db.get_session(session.id)
        assert restored == session
        assert [item.id for item in await db.list_sessions(session.working_directory)] == [session.id]

        message = user_message(
            session.id,
            "inspect image",
            [media_part(b"png", mime_type="image/png", detail="low")],
        )
        await db.save_message(message)
        conversation = await db.load_conversation(session.id)
        assert conversation.messages[0].role is Role.USER
        assert conversation.messages[0].parts[1].mime_type == "image/png"

        sequence = await db.next_sequence(session.id)
        await db.save_event(Event(sequence, "message_added", session.id, run_id, {"ok": True}))
        events = await db.load_events(session.id)
        assert [(event.sequence, event.run_id, event.data["ok"]) for event in events] == [
            (sequence, run_id, True)
        ]
        started = await db.next_sequence(session.id)
        await db.save_event(
            Event(
                started,
                "tool_started",
                session.id,
                run_id,
                {"call_id": "call-1", "name": "read_file", "arguments": {"path": "a.txt"}},
            )
        )
        finished = await db.next_sequence(session.id)
        await db.save_event(
            Event(
                finished,
                "tool_finished",
                session.id,
                run_id,
                {"call_id": "call-1", "name": "read_file", "success": True, "output": "ok"},
            )
        )

        receipt = RunRecord(
            run_id=run_id,
            session_id=session.id,
            status="finished",
            turns=2,
            input_tokens=11,
            output_tokens=7,
            cached_tokens=3,
            reasoning_tokens=2,
            duration_seconds=0.25,
            cost_usd=0.01,
            model="integration-model",
            retries=1,
        )
        await db.save_run(receipt)
        assert await db.get_run(run_id) == receipt
        assert [item.run_id for item in await db.list_runs(session.id)] == [run_id]
        metrics = await db._fetchrow(
            """
            SELECT m.llm_calls, m.tool_calls, m.total_tokens
            FROM v_run_metrics m JOIN agent_runs ar ON ar.id=m.run_id
            WHERE ar.metadata->>'native_run_id'=$1
            """,
            run_id,
        )
        assert dict(metrics) == {"llm_calls": 2, "tool_calls": 1, "total_tokens": 18}

        grant = PermissionGrant("file_*", PermissionDuration.SESSION, session.id, "src/")
        await db.save_permission(grant)
        assert await db.load_permissions(session.id) == [grant]

        await db.save_memory(memory)
        assert [item.id for item in await db.load_memories(session.id)] == [memory.id]
        await db.touch_memory(memory.id, memory.last_used_at)

        assert await db.delete_session(session.id) is True
        assert await db.get_session(session.id) is None
        assert await db.load_events(session.id) == []
        assert await db.list_runs(session.id) == []
        assert await db.load_permissions(session.id) == []
        assert await db.load_memories(session.id) == []
    finally:
        await db.delete_session(session.id)
        await db.close()

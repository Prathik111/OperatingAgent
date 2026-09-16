from __future__ import annotations

import pytest
from api.config import ApiSettings
from api.repository.memory import InMemoryTaskRepository
from api.services.approval_gateway import ApprovalGateway
from api.services.event_broker import EventBroker
from api.services.task_service import TaskService
from common.agent import AgentRunResult
from common.enums import AgentTrack, RunStatus


class StaticOrchestrator:
    def __init__(self, output: str, status: RunStatus = RunStatus.COMPLETED) -> None:
        self._output = output
        self._status = status

    async def run(self, task, on_event=None):
        return AgentRunResult(
            status=self._status,
            output=self._output,
            duration_ms=5.0,
            llm_calls=1,
            tool_calls=0,
            total_tokens=10,
            cost=0.0,
            metadata={"error": "boom"} if self._status is not RunStatus.COMPLETED else {},
        )


class NativeReceiptOrchestrator:
    """Shaped like the native track: no llm_call events; usage on the receipt.

    The real native orchestrator reports tokens/cost/turns through
    AgentRunResult and its metadata, never through llm_call events, so the
    evaluation path must fall back to the receipt when the repository's
    llm_calls table is empty (whose in-memory sums are a fabricated 0).
    """

    async def run(self, task, on_event=None):
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output="Component Code\n```tsx\nexport const A = 1;\n```\n",
            duration_ms=1200.0,
            llm_calls=3,
            tool_calls=2,
            total_tokens=2100,
            cost=0.0123,
            metadata={
                "llm_calls": 3,
                "tool_calls": 2,
                "total_tokens": 2100,
                "cost": 0.0123,
            },
        )


class WorkspaceCaptureOrchestrator:
    def __init__(self) -> None:
        self.workspaces: list[str] = []

    async def run(self, task, on_event=None):
        self.workspaces.append(str(task.metadata.get("sandbox_workspace") or ""))
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output="hello",
            duration_ms=1.0,
            llm_calls=1,
            tool_calls=0,
            total_tokens=12,
            metadata={"total_tokens": 12},
        )


class TransientFailureOrchestrator:
    def __init__(self, failures: int, message: str) -> None:
        self.failures = failures
        self.message = message
        self.calls = 0

    async def run(self, task, on_event=None):
        self.calls += 1
        if self.calls <= self.failures:
            return AgentRunResult(
                status=RunStatus.FAILED,
                output=None,
                duration_ms=1.0,
                llm_calls=0,
                tool_calls=0,
                total_tokens=0,
                metadata={"error": self.message},
            )
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output="hello",
            duration_ms=1.0,
            llm_calls=1,
            tool_calls=0,
            total_tokens=5,
        )


def _service(output: str, status: RunStatus = RunStatus.COMPLETED):
    repository = InMemoryTaskRepository()
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: StaticOrchestrator(output, status)},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.NATIVE, sandbox_workspace="."),
        background=set(),
    )
    return service, repository


_CODE_CHECK = {
    "name": "component_code",
    "kind": "labeled_code_block",
    "labels": ["component code"],
}


async def test_start_evaluation_rejects_malformed_checks_before_any_run() -> None:
    service, _ = _service("hello")
    with pytest.raises(ValueError, match="unknown check kind"):
        await service.start_evaluation(
            name="bad",
            version="1",
            tracks=[AgentTrack.NATIVE],
            cases=[{"id": "c1", "goal": "g", "checks": [{"name": "x", "kind": "vibes"}]}],
        )
    dashboard = await service.evaluation_dashboard()
    assert dashboard["runs"] == 0


async def test_startup_recovery_closes_abandoned_evaluation_run() -> None:
    service, repository = _service("hello")
    record = await repository.create_evaluation_run(
        "abandoned",
        "1",
        AgentTrack.NATIVE.value,
        [{"id": "c1", "goal": "hello"}],
    )

    before = await service.evaluation_dashboard()
    assert next(run for run in before["run_breakdown"] if run["id"] == record["id"])["status"] == "running"

    await service.recover_stale_executions()

    after = await service.evaluation_dashboard()
    recovered = next(run for run in after["run_breakdown"] if run["id"] == record["id"])
    assert recovered["status"] == "completed"
    assert recovered["finished_at"] is not None


async def test_dashboard_handles_unjudged_runs_without_crashing() -> None:
    service, _repository = _service("hello")
    await service.start_evaluation(
        name="deterministic-only",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "say hello", "expected_output_contains": "hello"}],
    )
    await service.wait_idle()
    dashboard = await service.evaluation_dashboard()
    assert dashboard["judge_average"] is None
    assert dashboard["judge_judged"] == 0
    assert dashboard["available"] is True


async def test_evaluation_mount_workspace_is_normalized_and_passed_to_agent() -> None:
    from pathlib import Path

    repository = InMemoryTaskRepository()
    orchestrator = WorkspaceCaptureOrchestrator()
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.NATIVE, sandbox_workspace="."),
        background=set(),
    )
    await service.start_evaluation(
        name="mount",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "hello", "working_directory": "."}],
    )
    await service.wait_idle()
    assert orchestrator.workspaces == [str(Path.cwd().resolve())]


async def test_evaluation_retries_rate_limit_then_scores_success(monkeypatch) -> None:
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("api.services.task_service.asyncio.sleep", no_wait)
    repository = InMemoryTaskRepository()
    orchestrator = TransientFailureOrchestrator(1, "429 too many requests")
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(
            default_track=AgentTrack.NATIVE,
            sandbox_workspace=".",
            execution_retry_attempts=2,
        ),
        background=set(),
    )
    await service.start_evaluation(
        name="retry",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "say hello", "expected_output_contains": "hello"}],
    )
    await service.wait_idle()

    dashboard = await service.evaluation_dashboard()
    assert orchestrator.calls == 2
    assert dashboard["results"] == 1
    assert dashboard["excluded_results"] == 0
    assert dashboard["pass_rate"] == 1.0


async def test_exhausted_rate_limit_is_visible_but_not_scored(monkeypatch) -> None:
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("api.services.task_service.asyncio.sleep", no_wait)
    repository = InMemoryTaskRepository()
    orchestrator = TransientFailureOrchestrator(10, "rate limit reached (429)")
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(
            default_track=AgentTrack.NATIVE,
            sandbox_workspace=".",
            execution_retry_attempts=2,
        ),
        background=set(),
    )
    await service.start_evaluation(
        name="excluded",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "say hello", "expected_output_contains": "hello"}],
    )
    await service.wait_idle()

    dashboard = await service.evaluation_dashboard()
    assert orchestrator.calls == 3
    assert dashboard["results"] == 0
    assert dashboard["excluded_results"] == 1
    assert dashboard["rate_limited_results"] == 1
    assert dashboard["pass_rate"] is None
    run = dashboard["comparison"][0]
    assert run["result_count"] == 0
    assert run["excluded_count"] == 1
    execution = dashboard["executions"][0]
    assert execution["outcome"] == "rate_limited"
    assert execution["excluded"] is True


async def test_connection_failure_is_retried_and_excluded_as_provider_unavailable(monkeypatch) -> None:
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("api.services.task_service.asyncio.sleep", no_wait)
    repository = InMemoryTaskRepository()
    orchestrator = TransientFailureOrchestrator(
        10, "planner failed with provider ollama: All connection attempts failed"
    )
    service = TaskService(
        orchestrators={AgentTrack.LANGGRAPH: orchestrator},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(
            default_track=AgentTrack.LANGGRAPH,
            sandbox_workspace=".",
            execution_retry_attempts=1,
        ),
        background=set(),
    )
    await service.start_evaluation(
        name="offline",
        version="1",
        tracks=[AgentTrack.LANGGRAPH],
        cases=[{"id": "c1", "goal": "say hello"}],
    )
    await service.wait_idle()

    dashboard = await service.evaluation_dashboard()
    assert orchestrator.calls == 2
    assert dashboard["results"] == 0
    assert dashboard["excluded_results"] == 1
    assert dashboard["rate_limited_results"] == 0
    assert dashboard["executions"][0]["outcome"] == "provider_unavailable"


async def test_evaluation_saves_per_check_scores() -> None:
    service, repository = _service("Component Code\n```tsx\nexport const A = 1;\n```\n")
    await service.start_evaluation(
        name="checked",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "build it", "checks": [_CODE_CHECK]}],
    )
    await service.wait_idle()
    evaluation = next(iter(repository._evaluations.values()))
    result = evaluation["results"][0]
    assert result["success"] is True
    metrics = {score["metric"]: score["value"] for score in result["scores"]}
    assert metrics["correctness"] == 1.0
    assert metrics["check.component_code"] == 1.0


async def test_evaluation_failure_reason_names_the_failed_check() -> None:
    service, repository = _service("no labeled sections at all")
    await service.start_evaluation(
        name="checked",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "build it", "checks": [_CODE_CHECK]}],
    )
    await service.wait_idle()
    evaluation = next(iter(repository._evaluations.values()))
    result = evaluation["results"][0]
    assert result["success"] is False
    assert "component_code" in (result["failure_reason"] or "")


async def test_failed_run_failure_reason_keeps_the_run_error() -> None:
    service, repository = _service("irrelevant output", status=RunStatus.FAILED)
    await service.start_evaluation(
        name="checked",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "build it", "checks": [_CODE_CHECK]}],
    )
    await service.wait_idle()
    evaluation = next(iter(repository._evaluations.values()))
    result = evaluation["results"][0]
    assert result["success"] is False
    # The real run error must survive, not be flattened to "run failed".
    assert "boom" in (result["failure_reason"] or "")


async def test_native_run_tokens_come_from_the_receipt() -> None:
    # Regression: with no llm_call rows the in-memory get_run_metrics sums to
    # 0 (not None), so the old `is None` gap check never fired and native
    # evaluations showed zero tokens/cost despite a receipt that knew better.
    repository = InMemoryTaskRepository()
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: NativeReceiptOrchestrator()},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.NATIVE, sandbox_workspace="."),
        background=set(),
    )
    await service.start_evaluation(
        name="tokens",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "c1", "goal": "build it", "checks": [_CODE_CHECK]}],
    )
    await service.wait_idle()

    evaluation = next(iter(repository._evaluations.values()))
    result = evaluation["results"][0]
    metrics = {score["metric"]: score["value"] for score in result["scores"]}
    assert metrics["tokens"] == 2100
    assert metrics["cost"] == pytest.approx(0.0123)
    assert metrics["latency"] == pytest.approx(1200.0)
    dashboard = await service.evaluation_dashboard()
    assert dashboard["total_tokens"] == 2100
    assert dashboard["total_cost"] == pytest.approx(0.0123)

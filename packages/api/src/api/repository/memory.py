"""In-memory ``TaskRepository`` — the default, fully hermetic backend.

Plain dicts, no I/O. It is the store the unit suite runs against and the
sensible default for local dev where a Postgres instance is overkill. It keeps
the same task/run/event shape as the Postgres store so switching backends
changes nothing above the repository seam.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from common.agent import AgentRunResult, AgentTask
from common.approvals import ApprovalRecord, ApprovalRequest
from common.config import AgentConfig
from common.enums import RunStatus, TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord

from ..errors import TaskNotFound, ThreadNotFound
from .base import OPEN_RUN_STATUSES, OpenRun, RunSummary, ThreadRecord


def _utc(value: datetime) -> datetime:
    """Normalize legacy naive UTC timestamps before comparison."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(slots=True)
class _Run:
    id: str
    task_id: str
    status: RunStatus
    order: int
    output: str | None = None
    last_error: str | None = None
    events: list[tuple[int, str, dict]] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    trace_refs: list[dict] = field(default_factory=list)
    approvals: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    finished_at: datetime | None = None


@dataclass(slots=True)
class _Thread:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class InMemoryTaskRepository:
    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._threads: dict[str, _Thread] = {}
        self._task_status: dict[str, TaskStatus] = {}
        self._runs: dict[str, _Run] = {}
        self._approvals: dict[str, ApprovalRecord] = {}
        self._order = itertools.count()
        self._tools: dict[tuple[str, str], tuple[str, dict]] = {}
        self._evaluations: dict[str, dict] = {}

    async def get_evaluation_dashboard(self) -> dict:
        runs = list(self._evaluations.values())
        all_results = [result for run in runs for result in run["results"]]

        def _excluded_kind(result: dict) -> str | None:
            marker = next(
                (
                    str(score.get("metric", "")).removeprefix("excluded.")
                    for score in result["scores"]
                    if str(score.get("metric", "")).startswith("excluded.")
                ),
                None,
            )
            return marker or None

        results = [result for result in all_results if _excluded_kind(result) is None]
        scores = [score for result in results for score in result["scores"] if score.get("value") is not None]
        correctness = [float(score["value"]) for score in scores if score.get("metric") == "correctness"]
        metrics: dict[str, list[float]] = {}
        for item in scores:
            metrics.setdefault(item["metric"], []).append(float(item["value"]))
        passed = sum(1 for result in results if result["success"])
        tool_rates = [float(r["tool_success_rate"]) for r in results if r.get("tool_success_rate") is not None]
        unique_cases = {(run["suite_id"], case_id) for run in runs for case_id in run["case_ids"]}

        def _judge_stats(run_results):
            scores = [float(s["value"]) for r in run_results for s in r["scores"] if s.get("metric") == "judge.overall" and s.get("value") is not None]
            errors = sum(1 for r in run_results for s in r["scores"] if s.get("metric") == "judge.error")
            return (sum(scores) / len(scores)) if scores else None, len(scores), errors

        judge_stats = {
            run["id"]: _judge_stats(
                [result for result in run["results"] if _excluded_kind(result) is None]
            )
            for run in runs
        }
        judged_runs = [stats for stats in judge_stats.values() if stats[0] is not None and stats[1] > 0]
        active_metrics: dict[str, dict[str, float]] = {}
        for evaluation in runs:
            if evaluation.get("finished_at"):
                continue
            aggregate = {"latency_ms": 0.0, "tokens": 0.0, "cost": 0.0, "runs": 0.0}
            for task in self._tasks.values():
                if str(task.metadata.get("evaluation_run_id") or "") != str(evaluation["id"]):
                    continue
                latest = await self.get_latest_run(task.id)
                if latest is None or latest.status not in OPEN_RUN_STATUSES:
                    continue
                metrics = await self.get_run_metrics(latest.run_id)
                aggregate["runs"] += 1
                aggregate["latency_ms"] += float(metrics.get("latency_ms") or 0)
                aggregate["tokens"] += float(metrics.get("total_tokens") or 0)
                aggregate["cost"] += float(metrics.get("cost") or 0)
            active_metrics[evaluation["id"]] = aggregate
        run_breakdown = []
        for run in runs:
            scored_results = [
                result for result in run["results"] if _excluded_kind(result) is None
            ]
            excluded_results = [
                result for result in run["results"] if _excluded_kind(result) is not None
            ]
            correctness_scores = [
                float(score["value"])
                for result in scored_results
                for score in result["scores"]
                if score.get("metric") == "correctness"
                and score.get("value") is not None
            ]
            run_tool_rates = [
                float(result["tool_success_rate"])
                for result in scored_results
                if result.get("tool_success_rate") is not None
            ]
            run_breakdown.append({
                "id": run["id"],
                "suite": f'{run["name"]} v{run["version"]}',
                "track": run["track"],
                "started_at": run["started_at"],
                "finished_at": run.get("finished_at"),
                "result_count": len(scored_results),
                "excluded_count": len(excluded_results),
                "rate_limited_count": sum(
                    1 for result in excluded_results
                    if _excluded_kind(result) == "rate_limited"
                ),
                "passed_count": sum(1 for result in scored_results if result["success"]),
                "pass_rate": (
                    sum(1 for result in scored_results if result["success"])
                    / len(scored_results)
                ) if scored_results else None,
                "average_score": (
                    sum(correctness_scores) / len(correctness_scores)
                ) if correctness_scores else None,
                "avg_latency_ms": (
                    sum(float(result.get("latency_ms", 0) or 0) for result in scored_results)
                    / len(scored_results)
                ) if scored_results else None,
                "total_tokens": sum(
                    int(result.get("total_tokens", 0) or 0) for result in scored_results
                ),
                "total_cost": sum(
                    float(result.get("cost", 0) or 0) for result in scored_results
                ),
                "tool_success_rate": (
                    sum(run_tool_rates) / len(run_tool_rates)
                ) if run_tool_rates else None,
                "judge_average": judge_stats[run["id"]][0],
                "judge_judged": judge_stats[run["id"]][1],
                "judge_errors": judge_stats[run["id"]][2],
                "status": "completed" if run.get("finished_at") else "running",
            })
        for summary in run_breakdown:
            active = active_metrics.get(summary["id"])
            if not active or not active["runs"]:
                continue
            completed_count = summary["result_count"]
            active_count = int(active["runs"])
            summary["total_tokens"] = int(summary["total_tokens"] or 0) + int(active["tokens"])
            summary["total_cost"] = float(summary["total_cost"] or 0) + active["cost"]
            summary["avg_latency_ms"] = (
                (float(summary["avg_latency_ms"] or 0) * completed_count + active["latency_ms"])
                / max(1, completed_count + active_count)
            )
        comparison = []
        for track in ("native", "langgraph"):
            candidates = [run for run in run_breakdown if run["track"] == track]
            if not candidates:
                continue
            latest = max(candidates, key=lambda run: run["started_at"])
            comparison.append({"track": track, **{key: latest.get(key) for key in ("id", "suite", "pass_rate", "average_score", "avg_latency_ms", "total_tokens", "total_cost", "tool_success_rate", "judge_average", "judge_judged", "judge_errors", "result_count", "excluded_count", "rate_limited_count", "status")}})
        executions = []
        for evaluation in sorted(runs, key=lambda run: run["started_at"], reverse=True):
            case_keys = {
                case_id: str(case.get("id") or case_id)
                for case_id, case in zip(evaluation["case_ids"], evaluation["cases"])
            }
            for result in evaluation["results"]:
                agent_run = self._runs.get(result["agent_run_id"])
                if agent_run is None:
                    continue
                task = self._tasks.get(agent_run.task_id)
                if task is None:
                    continue
                judge_row = next(
                    (s for s in result["scores"] if s.get("metric") == "judge.overall"), None
                )
                judge_error_row = next(
                    (s for s in result["scores"] if s.get("metric") == "judge.error"), None
                )
                executions.append({
                    "id": result["id"],
                    "evaluation_run_id": evaluation["id"],
                    "agent_run_id": agent_run.id,
                    "task_id": task.id,
                    "thread_id": task.thread_id,
                    "suite": f'{evaluation["name"]} v{evaluation["version"]}',
                    "track": evaluation["track"],
                    "case_id": case_keys.get(result["case_id"], result["case_id"]),
                    "goal": task.goal,
                    "workspace": str(task.metadata.get("workspace") or ""),
                    "status": agent_run.status.value,
                    "output": agent_run.output,
                    "error": agent_run.last_error,
                    "success": bool(result["success"]),
                    "excluded": _excluded_kind(result) is not None,
                    "outcome": _excluded_kind(result) or ("passed" if result["success"] else "failed"),
                    "judge_score": float(judge_row["value"]) if judge_row is not None and judge_row.get("value") is not None else None,
                    "judge_comment": str(judge_row.get("comment") or "") if judge_row is not None else "",
                    "judge_error": str(judge_error_row.get("comment") or "") if judge_error_row is not None else "",
                    "created_at": task.created_at.isoformat(),
                    "finished_at": getattr(agent_run, "finished_at", None).isoformat() if getattr(agent_run, "finished_at", None) else None,
                })
            # Evaluation tasks are persisted before their result row. Expose
            # those in-flight cases so the dashboard can render a live
            # execution card and seamlessly replace it with the final result.
            completed_case_ids = {str(result["case_id"]) for result in evaluation["results"]}
            for index, case in enumerate(evaluation.get("cases", [])):
                case_id = str(evaluation["case_ids"][index])
                if case_id in completed_case_ids:
                    continue
                expected_ids = {case_id, str(case.get("id") or "")}
                candidates = []
                for task in self._tasks.values():
                    metadata = task.metadata
                    if str(metadata.get("evaluation_run_id") or "") != str(evaluation["id"]):
                        continue
                    if str(metadata.get("evaluation_case_id") or "") not in expected_ids:
                        continue
                    if task.track.value != evaluation["track"]:
                        continue
                    run_ids = [run for run in self._runs.values() if run.task_id == task.id]
                    if run_ids:
                        candidates.append((task, max(run_ids, key=lambda run: run.order)))
                if not candidates:
                    continue
                task, agent_run = max(candidates, key=lambda item: _utc(item[0].created_at))
                finished_at = getattr(agent_run, "finished_at", None)
                if finished_at is not None or agent_run.status not in OPEN_RUN_STATUSES:
                    # A terminal run should have a result shortly; avoid
                    # showing a stale loading card if persistence lags.
                    continue
                executions.append({
                    "id": f"pending:{evaluation['id']}:{case_id}",
                    "evaluation_run_id": evaluation["id"],
                    "agent_run_id": agent_run.id,
                    "task_id": task.id,
                    "thread_id": task.thread_id,
                    "suite": f'{evaluation["name"]} v{evaluation["version"]}',
                    "track": evaluation["track"],
                    "case_id": str(case.get("id") or case_id),
                    "goal": task.goal,
                    "workspace": str(task.metadata.get("workspace") or ""),
                    "status": agent_run.status.value,
                    "output": agent_run.output,
                    "error": agent_run.last_error,
                    "success": False,
                    "loading": True,
                    "judge_score": None,
                    "judge_comment": "",
                    "judge_error": "",
                    "created_at": task.created_at.isoformat(),
                    "finished_at": None,
                })
        executions = executions[:100]
        return {
            "available": bool(runs),
            "reason": None if runs else "Run an evaluation to populate this dashboard.",
            "suites": len({run["suite_id"] for run in runs}),
            "cases": len(unique_cases),
            "runs": len(runs),
            "results": len(results),
            "excluded_results": len(all_results) - len(results),
            "rate_limited_results": sum(1 for result in all_results if _excluded_kind(result) == "rate_limited"),
            "pass_rate": passed / len(results) if results else None,
            "average_score": sum(correctness) / len(correctness) if correctness else None,
            "avg_latency_ms": sum(float(r.get("latency_ms", 0) or 0) for r in results) / len(results) if results else None,
            "total_tokens": sum(int(r.get("total_tokens", 0)) for r in results) if results else None,
            "total_cost": sum(float(r.get("cost", 0)) for r in results) if results else None,
            "tool_success_rate": sum(tool_rates) / len(tool_rates) if tool_rates else None,
            "judge_average": (sum(v[0] * v[1] for v in judged_runs) / sum(v[1] for v in judged_runs)) if judged_runs else None,
            "judge_judged": sum(v[1] for v in judge_stats.values()),
            "judge_errors": sum(v[2] for v in judge_stats.values()),
            "metrics": [{"metric": key, "average": sum(values) / len(values), "count": len(values)} for key, values in sorted(metrics.items())],
            "run_breakdown": run_breakdown,
            "comparison": comparison,
            "executions": executions,
            "sources": {"database": True, "langfuse": False},
        }

    async def create_evaluation_run(self, name: str, version: str, track: str, cases: list[dict]) -> dict:
        run_id, suite_id = str(uuid4()), str(uuid4())
        # Reuse the logical suite/case identity for repeated benchmark runs.
        # PostgreSQL enforces this with unique constraints; the memory/SQLite
        # backends mirror that behavior so the dashboard does not inflate the
        # suite and case totals every time the same comparison is rerun.
        existing = next(
            (run for run in self._evaluations.values()
             if run["name"] == name and run["version"] == version),
            None,
        )
        if existing is not None and len(existing.get("cases", [])) == len(cases):
            suite_id = existing["suite_id"]
            case_ids = list(existing["case_ids"])
        else:
            case_ids = [str(uuid4()) for _ in cases]
        self._evaluations[run_id] = {"id": run_id, "suite_id": suite_id, "name": name, "version": version, "track": track, "case_ids": case_ids, "cases": cases, "results": [], "started_at": datetime.now(UTC).isoformat(), "finished_at": None}
        return {"id": run_id, "suite_id": suite_id, "case_ids": case_ids}

    async def save_evaluation_result(self, evaluation_run_id: str, suite_id: str, case_id: str, agent_run_id: str, success: bool, failure_reason: str | None) -> str:
        result_id = str(uuid4())
        self._evaluations[evaluation_run_id]["results"].append({"id": result_id, "case_id": case_id, "agent_run_id": agent_run_id, "success": success, "failure_reason": failure_reason, "scores": []})
        return result_id

    async def save_evaluation_score(self, result_id: str, metric: str, value: float | None, unit: str | None = None, comment: str | None = None) -> None:
        for run in self._evaluations.values():
            for result in run["results"]:
                if result["id"] == result_id:
                    result["scores"].append({"metric": metric, "value": value, "unit": unit, "comment": comment})
                    if metric == "latency":
                        result["latency_ms"] = value
                    elif metric == "tokens":
                        result["total_tokens"] = value
                    elif metric == "cost":
                        result["cost"] = value
                    elif metric == "tool_success":
                        result["tool_success_rate"] = value
                    return

    async def finish_evaluation_run(self, evaluation_run_id: str) -> None:
        self._evaluations[evaluation_run_id]["finished_at"] = datetime.now(UTC).isoformat()

    async def finish_abandoned_evaluation_runs(self) -> list[str]:
        abandoned = [
            run_id
            for run_id, evaluation in self._evaluations.items()
            if not evaluation.get("finished_at")
        ]
        finished_at = datetime.now(UTC).isoformat()
        for run_id in abandoned:
            self._evaluations[run_id]["finished_at"] = finished_at
        return abandoned

    async def get_run_metrics(self, run_id: str) -> dict:
        run = self._runs[run_id]
        tool_total = len(run.tool_calls)
        succeeded = sum(1 for call in run.tool_calls if call.success is True)
        return {
            "latency_ms": run.metadata.get("duration_ms"),
            "llm_calls": len(run.llm_calls),
            "tool_calls": tool_total,
            "tool_calls_succeeded": succeeded,
            "tool_success_rate": succeeded / tool_total if tool_total else None,
            "total_tokens": sum((call.prompt_tokens or 0) + (call.completion_tokens or 0) for call in run.llm_calls),
            "cost": sum((float(call.cost) if call.cost is not None else 0.0) for call in run.llm_calls),
        }
    async def save_task(self, task: AgentTask) -> None:
        task.created_at = _utc(task.created_at)
        self._tasks[task.id] = task
        # Normalize legacy naive task timestamps before storing or comparing.
        created_at = task.created_at
        thread = self._threads.get(task.thread_id)
        if thread is None:
            self._threads[task.thread_id] = _Thread(
                id=task.thread_id,
                title=task.metadata.get("title"),
                created_at=created_at,
                updated_at=created_at,
            )
        else:
            thread.updated_at = _utc(thread.updated_at)
            thread.created_at = _utc(thread.created_at)
            thread.updated_at = max(thread.updated_at, created_at)
        self._task_status.setdefault(task.id, TaskStatus.PLANNING)

    async def create_thread(self, thread_id: str, title: str | None = None) -> ThreadRecord:
        now = datetime.now(UTC)
        thread = self._threads.get(thread_id)
        if thread is None:
            thread = _Thread(thread_id, title, now, now)
            self._threads[thread_id] = thread
        return ThreadRecord(thread.id, thread.title, 0, thread.created_at, thread.updated_at)

    async def delete_thread(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        task_ids = {task.id for task in self._tasks.values() if task.thread_id == thread_id}
        for task_id in task_ids:
            del self._tasks[task_id]
            self._task_status.pop(task_id, None)
        for run_id in [run.id for run in self._runs.values() if run.task_id in task_ids]:
            del self._runs[run_id]
        for approval_id in [
            approval_id
            for approval_id, record in self._approvals.items()
            if record.request.task_id in task_ids
        ]:
            del self._approvals[approval_id]
        del self._threads[thread_id]
        return True

    async def get_task(self, task_id: str) -> AgentTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFound(task_id) from None

    async def list_threads(self, *, limit: int, offset: int) -> list[ThreadRecord]:
        for thread in self._threads.values():
            thread.created_at = _utc(thread.created_at)
            thread.updated_at = _utc(thread.updated_at)
        threads = sorted(
            self._threads.values(),
            key=lambda thread: (thread.updated_at, thread.id),
            reverse=True,
        )
        selected = threads[offset : offset + limit]
        return [
            ThreadRecord(
                id=thread.id,
                title=thread.title,
                task_count=sum(
                    task.thread_id == thread.id for task in self._tasks.values()
                ),
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
            for thread in selected
        ]

    async def list_tasks_by_thread(
        self, thread_id: str, *, limit: int, offset: int
    ) -> list[tuple[AgentTask, RunStatus | None]]:
        if thread_id not in self._threads:
            raise ThreadNotFound(thread_id)
        tasks = [
            task for task in self._tasks.values() if task.thread_id == thread_id
        ]
        for task in tasks:
            task.created_at = _utc(task.created_at)
        tasks = sorted(
            tasks,
            key=lambda task: (task.created_at, task.id),
            reverse=True,
        )
        return [
            (task, self._latest_run_status(task.id))
            for task in tasks[offset : offset + limit]
        ]

    async def create_run(
        self, task_id: str, config: AgentConfig, metadata: dict | None = None
    ) -> str:
        run_id = str(uuid4())
        self._runs[run_id] = _Run(
            id=run_id,
            task_id=task_id,
            status=RunStatus.CREATED,
            order=next(self._order),
            metadata=metadata or {},
        )
        return run_id

    async def get_latest_run_id(self, task_id: str) -> str | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        return max(runs, key=lambda r: r.order).id

    async def get_latest_run_metadata(self, task_id: str) -> dict:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return {}
        return dict(max(runs, key=lambda r: r.order).metadata)

    async def get_latest_run(self, task_id: str) -> RunSummary | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        run = max(runs, key=lambda r: r.order)
        return RunSummary(
            run_id=run.id,
            status=run.status,
            output=run.output,
            error=run.last_error,
            metadata=dict(run.metadata),
        )

    async def list_open_runs(self) -> list[OpenRun]:
        return [
            OpenRun(
                task_id=run.task_id,
                run_id=run.id,
                status=run.status,
                metadata=dict(run.metadata),
            )
            for run in self._runs.values()
            if run.status in OPEN_RUN_STATUSES
        ]

    async def mark_run_running(self, run_id: str) -> None:
        self._runs[run_id].status = RunStatus.RUNNING

    async def append_event(
        self, run_id: str, event: AgentEvent, sequence_number: int
    ) -> None:
        self._runs[run_id].events.append((sequence_number, event.type, event.payload))

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        self._runs[run_id].llm_calls.append(record)

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        tool_id = await self.upsert_tool(
            record.server_name,
            record.base_url,
            {
                "name": record.tool_name,
                "description": record.description,
                "input_schema": record.input_schema,
            },
        )
        from dataclasses import replace
        self._runs[run_id].tool_calls.append(replace(record, tool_id=tool_id))

    async def save_phase(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].phases.append(value)
        return value["id"]

    async def close_phase(self, run_id: str, payload: dict) -> None:
        for phase in self._runs[run_id].phases:
            if phase["id"] == payload["phase_id"]:
                phase.update(payload)
                return

    async def save_plan(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].plans.append(value)
        return value["id"]

    async def save_finding(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].findings.append(value)
        return value["id"]

    async def save_verification(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].verifications.append(value)
        return value["id"]

    async def save_trace_ref(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].trace_refs.append(value)
        return value["id"]

    async def save_approval(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4()), "status": "pending"}
        self._runs[run_id].approvals.append(value)
        return value["id"]

    async def upsert_tool(
        self, server_name: str, base_url: str | None, tool_spec: dict
    ) -> str:
        key = (server_name, str(tool_spec["name"]))
        existing = self._tools.get(key)
        tool_id = existing[0] if existing else str(uuid4())
        self._tools[key] = (tool_id, {**tool_spec, "base_url": base_url})
        return tool_id

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        run = self._runs[run_id]
        run.status = result.status
        run.output = result.output
        run.last_error = result.metadata.get("error")
        run.metadata.update(result.metadata)
        run.metadata["duration_ms"] = result.duration_ms
        run.finished_at = datetime.now(UTC)

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        self._task_status[task_id] = status

    async def get_latest_run_status(self, task_id: str) -> RunStatus | None:
        return self._latest_run_status(task_id)

    async def list_events(
        self, task_id: str, *, latest_run_only: bool = True
    ) -> list[AgentEvent]:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if latest_run_only and runs:
            runs = [max(runs, key=lambda r: r.order)]
        events: list[AgentEvent] = []
        for run in sorted(runs, key=lambda r: r.order):
            events.extend(
                AgentEvent(type=event_type, payload=dict(payload))
                for _sequence, event_type, payload in sorted(
                    run.events, key=lambda event: event[0]
                )
            )
        return events

    async def list_thread_events(self, thread_id: str) -> list[tuple[str, AgentEvent]]:
        task_ids = {task.id for task in self._tasks.values() if task.thread_id == thread_id}
        runs = sorted(
            (run for run in self._runs.values() if run.task_id in task_ids),
            key=lambda run: run.order,
        )
        return [
            (run.task_id, AgentEvent(type=event_type, payload=dict(payload)))
            for run in runs
            for _sequence, event_type, payload in sorted(
                run.events, key=lambda event: event[0]
            )
        ]

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        self._approvals.setdefault(request.id, ApprovalRecord(request=request))

    async def resolve_approval(
        self,
        request_or_payload: str | dict,
        approved: bool | None = None,
        note: str | None = None,
    ) -> None:
        if isinstance(request_or_payload, dict):
            payload = request_or_payload
            for run in self._runs.values():
                for approval in run.approvals:
                    if approval["id"] == payload["approval_id"]:
                        approval.update(payload)
                        approval["status"] = (
                            "approved" if payload["approved"] else "denied"
                        )
                        break
            request_id = str(payload.get("approval_id", ""))
            approved = bool(payload.get("approved"))
            note = payload.get("note")
        else:
            request_id = request_or_payload
        record = self._approvals.get(request_id)
        if record is not None and approved is not None:
            self._approvals[request_id] = ApprovalRecord(
                request=record.request, approved=approved, note=note
            )

    async def get_approval_state(self, request_id: str) -> ApprovalRecord | None:
        return self._approvals.get(request_id)

    async def list_pending_approvals(self) -> list[ApprovalRequest]:
        return [
            record.request
            for record in self._approvals.values()
            if record.approved is None
        ]

    def _latest_run_status(self, task_id: str) -> RunStatus | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        return max(runs, key=lambda r: r.order).status

    # -- test/introspection helpers (not part of the Protocol) --------------

    def events_for(self, run_id: str) -> list[tuple[int, str, dict]]:
        return list(self._runs[run_id].events)

    def llm_calls_for(self, run_id: str) -> list[LLMCallRecord]:
        return list(self._runs[run_id].llm_calls)

    def tool_calls_for(self, run_id: str) -> list[ToolCallRecord]:
        return list(self._runs[run_id].tool_calls)

    def task_status(self, task_id: str) -> TaskStatus | None:
        return self._task_status.get(task_id)

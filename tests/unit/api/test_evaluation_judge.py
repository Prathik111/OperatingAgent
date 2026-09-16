from __future__ import annotations

import json

import pytest
from api.config import ApiSettings
from api.repository.memory import InMemoryTaskRepository
from api.services.approval_gateway import ApprovalGateway
from api.services.event_broker import EventBroker
from api.services.task_service import TaskService
from common.agent import AgentRunResult
from common.enums import AgentTrack, RunStatus
from common.llm_judging import LLMJudge, _parse_verdict

GOOD_OUTPUT = (
    "## Component Code\n```tsx\nexport function Card() {}\n```\n"
    "## Component Info\nInputs: title. States: idle. Accessibility: labelled.\n"
    "## Pacing\n1. 0ms mount\n2. 150ms fade\n"
    "## Script\n```\nHello, then it fades.\n```\n"
)

_CODE_CHECK = {
    "name": "component_code",
    "kind": "labeled_code_block",
    "labels": ["component code"],
}
_SCRIPT_CHECK = {
    "name": "script",
    "kind": "labeled_code_block",
    "labels": ["script"],
}


class StaticOrchestrator:
    def __init__(self, output: str) -> None:
        self._output = output

    async def run(self, task, on_event=None):
        return AgentRunResult(
            status=RunStatus.COMPLETED,
            output=self._output,
            duration_ms=5.0,
            llm_calls=1,
            tool_calls=0,
            total_tokens=10,
            cost=0.0,
        )


class ScriptedJudgeFactory:
    """Records every prompt and answers from a scripted per-case verdict map.

    Also counts how many judge instances were built: fairness rule 1 says one
    instance per evaluation request, shared by every track.
    """

    def __init__(self, verdicts_by_case: dict[str, str | Exception]) -> None:
        self.verdicts = verdicts_by_case
        self.prompts: list[str] = []
        self.instances = 0

    def __call__(self, model_name: str) -> LLMJudge:
        self.instances += 1
        factory = self

        async def complete(messages: list[dict[str, str]]) -> str:
            factory.prompts.append(
                "\n".join(message["content"] for message in messages)
            )
            # The case is identified by its goal, the only per-case signal the
            # blind prompt is allowed to carry.
            for goal_fragment, answer in factory.verdicts.items():
                if goal_fragment in factory.prompts[-1]:
                    if isinstance(answer, Exception):
                        raise answer
                    return answer
            return json.dumps({"criteria": []})

        return LLMJudge(complete=complete, model=model_name)


def _all_pass(names: list[str]) -> str:
    return json.dumps(
        {
            "criteria": [{"name": name, "pass": True, "reason": "ok"} for name in names],
            "overall_comment": "solid",
        }
    )


def _service(orchestrator, judge_factory=None) -> tuple[TaskService, InMemoryTaskRepository]:
    repository = InMemoryTaskRepository()
    service = TaskService(
        orchestrators={AgentTrack.NATIVE: orchestrator, AgentTrack.LANGGRAPH: orchestrator},
        repository=repository,
        broker=EventBroker(),
        approvals=ApprovalGateway(),
        settings=ApiSettings(default_track=AgentTrack.NATIVE, sandbox_workspace="."),
        background=set(),
        judge_factory=judge_factory,
    )
    return service, repository


_CASES = [
    {"id": "card", "goal": "produce the card bundle", "checks": [_CODE_CHECK, _SCRIPT_CHECK]},
    {"id": "table", "goal": "produce the table bundle", "checks": [_CODE_CHECK, _SCRIPT_CHECK]},
]


async def test_judge_scores_saved_per_case_and_criterion() -> None:
    factory = ScriptedJudgeFactory(
        {
            "card bundle": _all_pass(["component_code", "script"]),
            "table bundle": json.dumps(
                {
                    "criteria": [
                        {"name": "component_code", "pass": True, "reason": "ok"},
                        {"name": "script", "pass": False, "reason": "off-step"},
                    ],
                    "overall_comment": "mixed",
                }
            ),
        }
    )
    service, repository = _service(StaticOrchestrator(GOOD_OUTPUT), factory)

    await service.start_evaluation(
        name="judged",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=_CASES,
        judge_model="judge-x",
    )
    await service.wait_idle()

    evaluation = next(iter(repository._evaluations.values()))
    by_case = {}
    for result in evaluation["results"]:
        metrics = {score["metric"]: score["value"] for score in result["scores"]}
        assert metrics["correctness"] == 1.0  # judge never touches correctness
        by_case[evaluation["case_ids"][evaluation["results"].index(result)]] = metrics
    ordered = [
        metrics
        for metrics in by_case.values()
        if "judge.overall" in metrics
    ]
    assert sorted(m["judge.overall"] for m in ordered) == [0.5, 1.0]
    judged = next(m for m in ordered if m["judge.overall"] == 0.5)
    assert judged["judge.script"] == 0.0
    assert judged["judge.component_code"] == 1.0
    # The judge errored on nothing: no judge.error rows.
    assert all("judge.error" not in m for m in ordered)


async def test_one_judge_instance_shared_across_tracks() -> None:
    factory = ScriptedJudgeFactory(
        {
            "bundle": _all_pass(["component_code", "script"]),
        }
    )
    service, _ = _service(StaticOrchestrator(GOOD_OUTPUT), factory)

    await service.start_evaluation(
        name="shared",
        version="1",
        tracks=[AgentTrack.NATIVE, AgentTrack.LANGGRAPH],
        cases=_CASES,
        judge_model="judge-x",
    )
    await service.wait_idle()

    # Fairness rule 1: one instance per request, even with two tracks.
    assert factory.instances == 1
    # Fairness rule 2: the prompts never name a track.
    for prompt in factory.prompts:
        for leaked in ("native", "langgraph", "track"):
            assert leaked not in prompt.lower()


async def test_judge_error_is_fail_visible_not_fatal() -> None:
    factory = ScriptedJudgeFactory(
        {
            "card bundle": RuntimeError("judge provider down"),
            "table bundle": _all_pass(["component_code", "script"]),
        }
    )
    service, repository = _service(StaticOrchestrator(GOOD_OUTPUT), factory)

    await service.start_evaluation(
        name="errors",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=_CASES,
        judge_model="judge-x",
    )
    await service.wait_idle()

    evaluation = next(iter(repository._evaluations.values()))
    results = evaluation["results"]
    # Both cases completed: a judge failure never fails the run.
    agent_runs = [repository._runs[r["agent_run_id"]] for r in results]
    assert all(run.status is RunStatus.COMPLETED for run in agent_runs)
    assert all(r["success"] for r in results)
    error_rows = [
        score
        for r in results
        for score in r["scores"]
        if score["metric"] == "judge.error"
    ]
    assert len(error_rows) == 1
    assert "judge provider down" in (error_rows[0]["comment"] or "")

    dashboard = await service.evaluation_dashboard()
    assert dashboard["judge_judged"] == 1
    assert dashboard["judge_errors"] == 1
    assert dashboard["judge_average"] == pytest.approx(1.0)


async def test_judge_skipped_for_incomplete_runs() -> None:
    class FailingOrchestrator:
        async def run(self, task, on_event=None):
            return AgentRunResult(
                status=RunStatus.FAILED,
                output=None,
                duration_ms=1.0,
                llm_calls=0,
                tool_calls=0,
                total_tokens=0,
                metadata={"error": "boom"},
            )

    factory = ScriptedJudgeFactory({})
    service, repository = _service(FailingOrchestrator(), factory)

    await service.start_evaluation(
        name="failed",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=_CASES,
        judge_model="judge-x",
    )
    await service.wait_idle()

    evaluation = next(iter(repository._evaluations.values()))
    judge_metrics = [
        score["metric"]
        for r in evaluation["results"]
        for score in r["scores"]
        if score["metric"].startswith("judge.")
    ]
    assert judge_metrics == []


async def test_judge_scores_case_without_deterministic_checks() -> None:
    factory = ScriptedJudgeFactory({"plain answer": _all_pass(["overall_quality"])})
    service, repository = _service(StaticOrchestrator("plain answer"), factory)

    await service.start_evaluation(
        name="judge-default",
        version="1",
        tracks=[AgentTrack.NATIVE],
        cases=[{"id": "plain", "goal": "return a plain answer"}],
        judge_model="judge-x",
    )
    await service.wait_idle()

    result = next(iter(repository._evaluations.values()))["results"][0]
    metrics = {score["metric"]: score["value"] for score in result["scores"]}
    assert metrics["judge.overall"] == 1.0
    assert metrics["judge.overall"] is not None


async def test_judge_model_without_factory_is_rejected_up_front() -> None:
    service, _ = _service(StaticOrchestrator(GOOD_OUTPUT), judge_factory=None)
    with pytest.raises(ValueError, match="no judge-capable model registry"):
        await service.start_evaluation(
            name="no-judge",
            version="1",
            tracks=[AgentTrack.NATIVE],
            cases=_CASES,
            judge_model="anything",
        )


async def test_unknown_judge_model_is_rejected_up_front() -> None:
    def factory(model_name: str) -> LLMJudge:
        raise KeyError(
            "judge model 'nope' is not registered (available: judge-x)"
        )

    service, _ = _service(StaticOrchestrator(GOOD_OUTPUT), factory)
    with pytest.raises(ValueError, match="not registered"):
        await service.start_evaluation(
            name="bad-model",
            version="1",
            tracks=[AgentTrack.NATIVE],
            cases=_CASES,
            judge_model="nope",
        )


async def test_dashboard_exposes_judge_stats_per_track() -> None:
    factory = ScriptedJudgeFactory({"bundle": _all_pass(["component_code", "script"])})
    service, _ = _service(StaticOrchestrator(GOOD_OUTPUT), factory)

    await service.start_evaluation(
        name="dashboard",
        version="1",
        tracks=[AgentTrack.NATIVE, AgentTrack.LANGGRAPH],
        cases=_CASES,
        judge_model="judge-x",
    )
    await service.wait_idle()

    dashboard = await service.evaluation_dashboard()
    assert dashboard["judge_judged"] == 4  # 2 cases x 2 tracks
    assert dashboard["judge_errors"] == 0
    assert dashboard["judge_average"] == pytest.approx(1.0)
    for entry in dashboard["comparison"]:
        assert entry["judge_average"] == pytest.approx(1.0)
        assert entry["judge_judged"] == 2
    # Every execution carries its own verdict so the UI can show per-case
    # judge results, not just aggregates.
    assert dashboard["executions"], "executions must be populated"
    for execution in dashboard["executions"]:
        assert execution["judge_score"] == pytest.approx(1.0)
        assert execution["judge_score"] is not None
        assert "solid" in (execution["judge_comment"] or "")
        assert execution["judge_error"] == ""
    # The metric breakdown surfaces every judge criterion automatically.
    metric_names = {row["metric"] for row in dashboard["metrics"]}
    assert {"judge.overall", "judge.script", "judge.component_code"} <= metric_names


def test_judge_rejects_duplicate_criteria() -> None:
    verdict = _parse_verdict(
        json.dumps({
            "criteria": [
                {"name": "component_code", "pass": True},
                {"name": "component_code", "pass": False},
            ]
        }),
        model="judge-x",
        expected_names=("component_code",),
    )
    assert verdict.status == "error"
    assert "duplicate criteria" in verdict.error


def test_judge_rejects_string_boolean_criteria() -> None:
    verdict = _parse_verdict(
        json.dumps({"criteria": [{"name": "component_code", "pass": "false"}]}),
        model="judge-x",
        expected_names=("component_code",),
    )
    assert verdict.status == "error"
    assert "non-boolean" in verdict.error

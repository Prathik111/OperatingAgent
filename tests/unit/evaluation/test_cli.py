from __future__ import annotations

import json
from pathlib import Path

from evaluation import _resolve_judge_suite
from evaluation.suite import default_suite, save_suite


def test_explicit_suite_path_always_wins(tmp_path: Path) -> None:
    suite_path = tmp_path / "custom-suite.json"
    save_suite(default_suite(), suite_path)
    suite = _resolve_judge_suite({"suite": "default"}, suite_path)
    assert suite is not None
    assert [case.id for case in suite.cases] == [case.id for case in default_suite().cases]


def test_default_suite_results_file_resolves_without_suite_path() -> None:
    payload = {"suite": "default", "results": []}
    suite = _resolve_judge_suite(payload, None)
    assert suite is not None
    assert suite.id == "default"
    assert suite.cases == default_suite().cases


def test_named_non_default_suite_without_path_resolves_to_none() -> None:
    assert _resolve_judge_suite({"suite": "my-suite"}, None) is None
    assert _resolve_judge_suite([{"case_id": "x"}], None) is None


def test_judge_results_row_with_null_metadata_is_tolerated(tmp_path: Path) -> None:
    # A null metadata field used to crash the re-judging loop with
    # AttributeError before the goal fallback was consulted.
    from evaluation import EvaluationResult

    row = EvaluationResult.from_dict(
        json.loads(
            json.dumps(
                {
                    "case_id": "x",
                    "track": "native",
                    "passed": True,
                    "output": "o",
                    "metadata": None,
                }
            )
        )
    )
    assert row.metadata == {}

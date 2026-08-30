"""The `agent-native runs` history view: what it lists and what it totals.

Offline by construction - it drives a `MemoryDatabase` directly, so nothing from
the postgres extra is needed. It checks the read surface the CLI is built on:

  * `list_sessions` filters by folder and returns newest first, with a limit;
  * the view picks the right runs for a session, a folder, or everything, keeping
    newest-first order and honouring the cap;
  * the totals line sums the receipt columns and matches the rows exactly - which
    is the plan's verify for this step ("totals matching the runs rows").

Run under pytest, or straight (the __main__ block) on a box without pytest:
    PYTHONPATH=packages/agent-native/src python3 packages/agent-native/tests/test_run_history.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent_native.conversation import Session
from agent_native.database import MemoryDatabase
from agent_native.loop import RunRecord
from agent_native.main import _list_runs_for_view, _render_runs_table, _runs_totals


def _folder(name: str) -> str:
    """A folder path resolved exactly the way the view resolves --dir, so a stored
    `working_directory` and a `--dir` value compare equal without a real folder."""
    return str(Path(f"/tmp/agent-native-{name}").expanduser().resolve())


async def _seed() -> tuple:
    """Two sessions in two folders and three runs: two under folder A, one under B.

    Saved oldest-first (r1, r2, r3), so a correct newest-first read gives r3, r2, r1.
    """
    db = MemoryDatabase()
    dir_a, dir_b = _folder("a"), _folder("b")
    await db.create_session(Session(agent="build", working_directory=dir_a, id="sa"))
    await db.create_session(Session(agent="build", working_directory=dir_b, id="sb"))

    await db.save_run(
        RunRecord(
            run_id="r1", session_id="sa", status="finished", turns=2,
            input_tokens=100, output_tokens=20, cost_usd=0.001, duration_seconds=1.0,
        )
    )
    await db.save_run(
        RunRecord(
            run_id="r2", session_id="sb", status="finished", turns=3,
            input_tokens=200, output_tokens=40, cost_usd=0.002, duration_seconds=2.0,
        )
    )
    await db.save_run(
        RunRecord(
            run_id="r3", session_id="sa", status="limit_reached", turns=5,
            input_tokens=300, output_tokens=60, cached_tokens=10, cost_usd=0.003,
            duration_seconds=3.0, retries=1, model="llama-3.3-70b",
        )
    )
    return db, dir_a, dir_b


def _args(session: str = "", directory: str = "", limit: int = 0) -> argparse.Namespace:
    return argparse.Namespace(session=session, dir=directory, limit=limit)


async def test_list_sessions_filters_by_folder_and_is_newest_first() -> None:
    db, dir_a, dir_b = await _seed()

    every = await db.list_sessions()
    assert [s.id for s in every] == ["sb", "sa"]        # sb created last -> first

    just_a = await db.list_sessions(working_directory=dir_a)
    assert [s.id for s in just_a] == ["sa"]             # folder narrows the set

    assert [s.id for s in await db.list_sessions(limit=1)] == ["sb"]   # cap keeps newest


async def test_view_all_lists_every_run_newest_first() -> None:
    db, _, _ = await _seed()
    runs = await _list_runs_for_view(db, _args())
    assert [r.run_id for r in runs] == ["r3", "r2", "r1"]


async def test_view_by_session_narrows_to_that_session() -> None:
    db, _, _ = await _seed()
    runs = await _list_runs_for_view(db, _args(session="sa"))
    assert [r.run_id for r in runs] == ["r3", "r1"]     # sa's two, newest first


async def test_view_by_folder_narrows_to_that_folders_sessions() -> None:
    db, dir_a, _ = await _seed()
    runs = await _list_runs_for_view(db, _args(directory=dir_a))
    assert [r.run_id for r in runs] == ["r3", "r1"]     # folder A holds session sa


async def test_session_wins_over_folder_when_both_given() -> None:
    db, _, dir_b = await _seed()
    # --session sa is the narrowest; --dir (folder B) is ignored, so we get sa's runs.
    runs = await _list_runs_for_view(db, _args(session="sa", directory=dir_b))
    assert [r.run_id for r in runs] == ["r3", "r1"]


async def test_limit_caps_to_the_most_recent() -> None:
    db, _, _ = await _seed()
    runs = await _list_runs_for_view(db, _args(limit=2))
    assert [r.run_id for r in runs] == ["r3", "r2"]


async def test_totals_match_the_rows() -> None:
    """The plan's verify: the totals equal the sum of the listed runs' receipts."""
    db, _, _ = await _seed()
    runs = await _list_runs_for_view(db, _args())
    totals = _runs_totals(runs)

    assert totals["runs"] == 3
    assert totals["turns"] == 2 + 3 + 5
    assert totals["input_tokens"] == 100 + 200 + 300
    assert totals["output_tokens"] == 20 + 40 + 60
    assert totals["cached_tokens"] == 10
    assert abs(totals["cost_usd"] - (0.001 + 0.002 + 0.003)) < 1e-9
    assert abs(totals["duration_seconds"] - 6.0) < 1e-9


async def test_render_has_a_row_per_run_and_a_totals_line() -> None:
    db, _, _ = await _seed()
    runs = await _list_runs_for_view(db, _args())
    table = _render_runs_table(runs)
    lines = table.splitlines()

    header = lines[0]
    for column in ("RUN", "STATUS", "TURNS", "IN", "OUT", "COST", "TIME", "MODEL"):
        assert column in header
    # r3 carried cached tokens and a retry, so those columns are shown.
    assert "CACHED" in header and "RETRIES" in header

    assert all(any(rid in line for line in lines) for rid in ("r1", "r2", "r3"))

    totals_line = next(line for line in lines if line.startswith("TOTALS"))
    assert "TOTALS (3 runs)" in totals_line
    # the summed numbers show up on that line, formatted like the per-run receipt
    for token in ("10", "600", "120", "0.006000", "6.00s"):
        assert token in totals_line


# ---------------------------------------------------------------------------
# A plain-stdlib runner, so this file verifies on a box without pytest.
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        test_list_sessions_filters_by_folder_and_is_newest_first,
        test_view_all_lists_every_run_newest_first,
        test_view_by_session_narrows_to_that_session,
        test_view_by_folder_narrows_to_that_folders_sessions,
        test_session_wins_over_folder_when_both_given,
        test_limit_caps_to_the_most_recent,
        test_totals_match_the_rows,
        test_render_has_a_row_per_run_and_a_totals_line,
    ]
    failures: list = []
    for test in tests:
        try:
            asyncio.run(test())
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAIL - run history:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"PASS - run history: {len(tests)} tests "
          "(list_sessions, session/folder/all views, limit, totals match rows).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

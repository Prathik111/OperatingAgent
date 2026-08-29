# The comparison report

This folder holds the artifact that makes the thesis *legible instead of asserted*:
a side-by-side reading of the native agent and the LangGraph agent over the same
[evaluation suite](../evaluation/README.md). It is a consumer over data the runs
already produced — no new instrumentation, just the receipts and the event streams
lined up honestly.

## What's here

- `results/` — one JSON record per run, written by `evaluation run` (keyed by run
  id, so a run is never overwritten).
- `comparison.md` — the rendered report, written by `evaluation compare`.

Neither is committed with real thesis numbers yet: the LangGraph track is a
separate contributor's package (`packages/agent-langgraph`), still a stub, and the
native numbers should be generated on a real machine with a key rather than
fabricated here. The excerpt below is **illustrative — the numbers are made up** —
to show the shape of the output.

## What the report shows

`evaluation compare` reads the newest run per track from `results/` and renders, in
one Markdown file:

- **Headline** — pass rate, deterministic-only pass rate, total cost, tokens,
  turns, average turns per task, and wall-clock, one column per track.
- **Deltas** (for a two-track head-to-head) — each headline number as `B − A`, with
  a legend saying which direction is better for each metric. The word only names
  the sign; it passes no verdict.
- **By category** and a **by-task pass/fail matrix**.
- **Per-task metrics** — the receipt for each task and track: `turns / tokens / cost`.
- **Per-task deltas** — turns, tokens and cost as `B − A`, task by task. This is the
  table the thesis leans on: not just *who* passed, but what each task *cost* each
  agent.
- **Where the tool sequences forked** — because every tool call is a numbered event,
  the exact order each agent called its tools is recoverable. This lists the tasks
  where the two orders differ — the clearest single view of *how* two agents solved
  the same task differently.
- **Where they differ** — the tasks one track passed and the other failed.

## Two honesty rules, enforced in code

1. **Compare like with like.** Reports built on different suite versions are refused,
   not silently averaged — a cross-version average is a number that looks meaningful
   and isn't.
2. **Keep the judge separate.** The headline pass rate always sits next to a
   deterministic-only pass rate, so the score that doesn't depend on a model's
   opinion is always visible.

## Regenerating it

For a cold-start walkthrough (clone → install → key → run → compare), see the
[reproduction runbook](reproducing.md). The short version:

```bash
# Produce a run record per track (native today; langgraph is a separate,
# not-yet-built contributor track):
evaluation run --track native

# Render the newest run of each track into comparison.md:
evaluation compare

# Or compare specific tracks, in a fixed left-to-right order:
evaluation compare --tracks native,langgraph --out docs/comparison/comparison.md
```

## Illustrative excerpt (made-up numbers)

```markdown
## Headline

| Metric | native | langgraph |
| --- | --- | --- |
| Pass rate | 15/16 (94%) | 14/16 (88%) |
| Total cost (USD) | 0.0147 | 0.0294 |
| Total tokens | 2,650 | 4,470 |
| Total turns | 41 | 58 |

## Per-task deltas (langgraph - native)

Positive means `langgraph` spent more than `native` on that task.

| Task | Category | d turns | d tokens | d cost (USD) |
| --- | --- | --- | --- | --- |
| read_config_port | single_file_read | +1 | +650 | +0.0040 |
| rename_function_everywhere | multi_file_edit | +2 | +800 | +0.0084 |

## Where the tool sequences forked

| Task | Category | native | langgraph |
| --- | --- | --- | --- |
| read_config_port | single_file_read | read_file | read_file -> read_file |
```

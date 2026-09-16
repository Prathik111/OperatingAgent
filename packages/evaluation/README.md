# evaluation

Reusable evaluation helpers for OperatingAgent.

The complete evaluation flow — suite definition, execution, deterministic and
LLM judging, score persistence, dashboards, and the enforced fairness rules —
is documented in [`docs/evaluation/README.md`](../../docs/evaluation/README.md).

This package provides:

- suite definitions for benchmark cases, with named deterministic checks per case
- a generic runner that executes a case through an injected track callback
- a deterministic output judge shared with the API (`common.judging`)
- an optional LLM judge for open-ended quality scoring (`evaluation.scoring`)
- a comparison renderer for side-by-side reporting

## Deterministic checks

Each case names its checks; a run passes only when the run completed, the
output is non-empty, and every check passes. Checks are structural and
reproduce byte-for-byte. Kinds:

| Kind | Passes when |
| --- | --- |
| `contains` | every term appears in the output (case-insensitive) |
| `any_contains` | at least one term appears |
| `regex` | the pattern matches anywhere |
| `labeled_code_block` | after a label, a fenced code block exists (optionally in given languages, containing given terms) |
| `labeled_contains` | every term appears after a label |
| `labeled_regex` | the pattern matches somewhere after a label |

`expected_output_contains` still works and is folded in as a `contains` check,
so old suite files behave exactly as before.

## Optional LLM judge

The judge core lives in `common.llm_judging` (shared with the API service, so
the desktop path and this harness enforce one fairness contract). CLI usage:

```bash
# Deterministic only (default):
uv run --package evaluation evaluation run --out evaluation-results.json

# With the LLM judge, one model shared by every track in the run:
uv run --package evaluation evaluation run --judge-model gpt-oss-120b --out evaluation-results.json

# Re-judge an existing results file without re-running:
uv run --package evaluation evaluation judge evaluation-results.json --judge-model gpt-oss-120b

# Compare two result files into a markdown report:
uv run --package evaluation evaluation compare native-results.json langgraph-results.json --out comparison.md
```

Fairness rules, enforced structurally in `scoring.py`:

1. **One judge for every track** — a single judge instance (one model, one
   prompt, temperature 0) is built once per run and reused for every track.
2. **Blind judging** — the prompt contains only the goal, the output, and the
   rubric. It never contains the track, case id, or run metadata.
3. **Separate reporting** — the judge score never overwrites the deterministic
   pass/fail; it is stored beside it and rendered in its own tables.
4. **Fail-visible** — a judge that errors, returns unparseable JSON, or does
   not score exactly the requested criteria records `status="error"` and a
   `None` score; errored verdicts are counted, never averaged in as passes.
5. **Judge overhead accounted separately** — judging tokens and cost are
   recorded on the verdict, not on the judged run.

The runner is intentionally callback-driven so it can be wired to the native or
LangGraph track without hard-coding orchestration details into the harness.

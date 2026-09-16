# The evaluation flow

This document follows one evaluation end to end: how a suite is defined, how
both agent tracks run it, how outputs are judged (deterministically and —
optionally — by an LLM), where every number in the dashboard comes from, and
which fairness rules are enforced in code rather than promised in prose.

```
suite (xlsx/csv/json)      ┌────────────────────────────────────────────┐
  │                        │                                            │
  ▼                        │                                            │
desktop import ──POST /evaluations/runs──► TaskService.start_evaluation │
  or CLI `evaluation run`  │        │              │                    │
                           │   one evaluation run per track             │
                           │        │              │                    │
                           │        ▼              ▼                    │
                           │   native agent    langgraph agent          │
                           │   (per case: isolated session, numbered    │
                           │    events, terminal receipt)               │
                           │        │              │                    │
                           │        ▼              ▼                    │
                           │   deterministic judge (common.judging)     │
                           │   + optional LLM judge (common.llm_judging)│
                           │        │                                   │
                           │        ▼                                   │
                           │   evaluation_results + evaluation_scores   │
                           └────────────────────────────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
          desktop dashboard                    CLI compare report
          (GET /evaluations/dashboard)         (evaluation compare)
```

## 1. The suite

A suite is a list of cases. Each case is a record with:

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique case id |
| `goal` | yes | The prompt both agents receive |
| `working_directory` | no | Workspace for the case (blank = the selected folder) |
| `expected_output_contains` | no | Legacy check: output must contain this text |
| `checks` | no | Named deterministic checks (see §4) |
| `metadata` | no | JSON object with extra run metadata |

Suites arrive three ways:

- **Spreadsheet upload** (`docs/evaluation/ui-suite.xlsx`): the desktop
  Evaluation view imports the first sheet. The `checks` column accepts a
  preset (`ui_bundle`) or a `+`-joined subset (`code+info+pacing+script`),
  expanded to exactly the checks the CLI harness uses.
- **API**: `POST /evaluations/runs` with a JSON body (same fields).
- **CLI**: `evaluation run --suite my-suite.json`, or the built-in default
  suite written by `evaluation suite`.

## 2. Starting a run

`start_evaluation` (API service or CLI runner):

1. Validates every case's checks **synchronously** — a malformed check fails
   the request (422/400), never a background run.
2. Creates **one evaluation run per selected track** over the same cases.
3. Optionally builds **one LLM judge instance** for the whole request (see §5)
   and hands the same instance to every track's execution.
4. Executes the tracks **sequentially** — native finishes before langgraph
   starts (or vice versa). Both agents and the shared judge draw on the same
   provider quota; running them in parallel would have them contend for the
   same rate limit, so whichever track happened to hit a 429 would fail cases
   the other passed. That is noise, not a comparison. Sequential execution
   gives both tracks identical conditions and halves peak tokens-per-minute.
5. Returns immediately (202); the tracks execute in the background. A run is
   only marked finished **after** its cases are scored — including any LLM
   judging — so a client watching `finished_at` sees true completion.

### Rate limits (429s) — the decision

Rate limits are treated as *the provider being busy*, never as a quality
verdict, at every layer:

- **Agent LLM calls** retry in the provider client (`LLM_MAX_RETRIES`, default
  7, exponential backoff — roughly 30s cumulative, enough to outlast a typical
  tokens-per-minute window on a free-tier key).
- **The judge** retries with backoff too, honouring the provider's own
  suggested wait when it states one (Groq's "Please try again in 25.4s").
  Only after the retries are exhausted does the case record `judge.error` —
  counted visibly, never folded into the score as a zero.
- **Tracks run sequentially** (above), so peak TPM pressure is one agent plus
  the judge, not two agents competing.

If quota remains a constraint, run fewer cases per run, or a smaller judge
model.

## 3. Execution

Per case, per track:

- A fresh, isolated session runs the case's goal (`thread_id` is unique per
  evaluation-run × case, so one case's state never leaks into the next).
- Every interesting thing the agent does is a **numbered event**, persisted
  before delivery: LLM calls, tool calls, phases, plans, findings,
  verifications. This is the run's authoritative history.
- The run ends with a **terminal receipt**: status, output, turns, tokens,
  cost, duration. Native reports provider usage on the receipt; LangGraph
  usage flows through persisted `llm_call` rows and/or its Langfuse trace,
  with the receipt as a gap-filler.

Both tracks' recovery budgets are configurable so the comparison is not
confused by structural defaults: the native loop's turn budget
(`AGENT_MAX_ITERATIONS`) and the graph's whole-run replan budget
(`AGENT_MAX_REPLANS`, default 3) can be set to comparable values.

## 4. Deterministic judging

A case **passes** when the run completed, the output is non-empty, and every
named check passes. Checks are structural and reproduce byte-for-byte:

| Kind | Passes when |
| --- | --- |
| `contains` | every term appears in the output (case-insensitive) |
| `any_contains` | at least one term appears |
| `regex` | the pattern matches anywhere |
| `labeled_code_block` | after a label, a fenced code block exists (optionally in given languages, containing given terms) |
| `labeled_contains` | every term appears after a label |
| `labeled_regex` | the pattern matches somewhere after a label |

The UI-bundle presets check the artifacts that must accompany generated
component code — `component_code` (any framework), `component_info` (inputs,
states, accessibility), `pacing` (an explicit `\d+\s*ms` duration), `script`
(the narration block) — each anchored to a labeled section, so a failure says
*which artifact* is missing.

Results are persisted as: `correctness` (1.0/0.0) and one
`check.<name>` score per named check.

## 5. The LLM judge (optional)

Enabled per run: a `judge_model` name (desktop field or API `judge_model`)
from the runtime's model registry — the same wired providers the agent uses.
The judge scores *quality* on top of the deterministic verdict, under five
fairness rules that are **enforced structurally** in `common.llm_judging`:

1. **One judge for every track.** A single judge instance — one model, one
   prompt template, temperature 0 — is built once per request, before any
   track executes, and reused for every case of every track being compared.
   No track can be judged by a different model or prompt.
2. **Blind judging.** The judge's prompt contains only the goal, the output,
   and the rubric. It never sees the track, the case id, or any run metadata,
   so it cannot favour a track it cannot identify.
3. **Separate reporting.** The judge score never touches the deterministic
   `correctness`. It is stored under its own metrics and rendered in its own
   tables/boxes, so the opinion-based score is always distinguishable from
   the reproducible one.
4. **Fail-visible, not fail-silent.** A judge that errors, returns
   unparseable JSON, or does not score *exactly* the requested criteria
   records `status="error"`. Errored judgements are counted (`judge.error`
   metric), never averaged in as passes or zeros.
5. **Judge overhead is accounted separately.** Provider-reported tokens and
   cost consumed by judging are recorded on the verdict itself, never on the
   judged run's metrics.

The rubric maps each named check to a quality criterion (e.g. `pacing` → "the
timing plan is concrete and usable: every step has an explicit duration…").
The verdict is a per-criterion pass/fail with one-line reasons plus an overall
comment.

Persisted per result:

| Metric | Meaning |
| --- | --- |
| `judge.overall` | fraction of criteria passed (0–1), or absent when errored |
| `judge.<criterion>` | 1.0/0.0 per criterion, reason in the comment |
| `judge.error` | present (value NULL) only when the judge failed for this case |

The dashboard aggregates these into `judge_average`, `judge_judged` and
`judge_errors` per track, shown beside — never merged into — the pass rate.
While a comparison is running, the desktop shows live progress (per track:
cases completed, cases judged, judge errors) and holds the *Compare agents*
button until every started run — judging included — is finished.

## 6. Scores and the dashboard

Every result carries: `correctness`, `check.*`, `latency`, `tokens`, `cost`,
`tool_success`, and (when judging) `judge.*`. `GET /evaluations/dashboard`
aggregates them: headline pass rate and averages, per-run breakdown, the
latest run per track for side-by-side comparison, and per-case execution
transcripts (prompt + response). The metric breakdown table includes every
`judge.*` criterion automatically.

## 7. The CLI harness

The same judge and checks are available without the API server:

```bash
# Built-in suite with the LLM judge, one judge shared by both tracks:
uv run --package evaluation evaluation run --judge-model gpt-oss-120b --out results.json

# Re-judge an existing results file without re-running the agents:
uv run --package evaluation evaluation judge results.json --judge-model gpt-oss-120b

# Render the side-by-side report (deterministic tables, then judge tables):
uv run --package evaluation evaluation compare a.json b.json --out comparison.md
```

The report keeps the same honesty rules: deterministic results first, judge
scores in their own section, judge deltas only over cases both tracks had
judged cleanly, with the judged/error counts always visible.

## 8. Honesty rules, enforced in code

1. **Compare like with like.** The CLI comparison refuses results over
   different case sets; suite version mismatches are rejected, not averaged.
2. **The judge stays separate.** Every judge surface — dashboard, report,
   scores table — keeps judge metrics disentangled from deterministic pass
   rates, and always shows how many cases were judged cleanly vs errored.
3. **Judge failures are visible.** An unavailable judge degrades to
   `judge.error` rows and a count, never to a fabricated score.

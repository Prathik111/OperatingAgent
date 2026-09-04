# Reproducing the thesis comparison

A cold-start guide: from a clean checkout to a rendered comparison report. The thesis compares a
hand-built agent (the *native* track) against a framework-built one (the *langgraph* track) over
one fixed [task suite](../evaluation/README.md); this runbook is the exact path to regenerate the
numbers rather than take them on faith.

Everything here is copy-pasteable. Where a step needs a secret or a network, it says so.

## What you get, and the one honest caveat

`evaluation run` produces a JSON record per run under `docs/comparison/results/`; `evaluation
compare` renders the newest run of each track into `docs/comparison/comparison.md`. The **native
track runs today** (it is this project). The **langgraph track is a separate contributor's
package** (`packages/agent-langgraph`), a stub at the time of writing — so a solo reader can
reproduce the *native* column now, and the full head-to-head once that track lands. The harness
measures both through the identical `Track` interface, so adding the second track changes no code
here.

## Prerequisites

- **uv** (the workspace's package manager). Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **A model.** Either a **Groq API key** (the default, cloud) or a local **Ollama** server (no
  key, no network). Pick one; both are shown below.
- Python is provisioned by uv from each package's `requires-python` — no separate install needed.

## 1. Get the code and install

```bash
git clone https://github.com/Prathik111/OperatingAgent.git
cd OperatingAgent

# Install the evaluation package with everything a live native run needs
# (agent-native + its MCP gateway). 'native' is the extra that pulls those in.
uv sync --package evaluation --extra native
```

The `evaluation` command is a console script (`evaluation.cli:main`). Run it through uv with
`uv run --package evaluation evaluation ...`.

## 2. See the suite without running anything

This needs no model and no key — a good check that the install worked:

```bash
uv run --package evaluation evaluation list
```

It prints the sixteen tasks and their categories (suite version `1.0`).

## 3. Provide a model

**Option A — Groq (default).** Put a key where the agent looks for it:

```bash
export GROQ_API_KEY=sk-...          # or a .env file at the repo root with GROQ_API_KEY=...
```

The native track's default Groq model is `gpt-oss-20b`; override with `--model`.

**Option B — Ollama (local, no key).** Start Ollama and pull the default local model, then pass
`--provider ollama`:

```bash
ollama serve &
ollama pull qwen3.5:4b-q4_K_M       # the native track's default ollama model
```

## 4. Run the native track over the suite

```bash
# Whole suite, Groq:
uv run --package evaluation evaluation run --track native

# ...or against a local model:
uv run --package evaluation evaluation run --track native --provider ollama

# A quick smoke run of two tasks:
uv run --package evaluation evaluation run --track native --tasks read_config_port,count_data_lines

# Tag the run with a git sha so a later compare can identify it:
uv run --package evaluation evaluation run --track native --label "$(git rev-parse --short HEAD)"
```

Each run writes a JSON record to `docs/comparison/results/`, keyed by the run id printed at the
end. Shell tools run on the machine by default (`--sandbox off`); pass `--sandbox on` to run them
in a container instead.

## 5. Render the comparison

```bash
# Newest run of every track found in results/:
uv run --package evaluation evaluation compare

# ...or fix the tracks and order explicitly:
uv run --package evaluation evaluation compare --tracks native,langgraph --out docs/comparison/comparison.md
```

The report gives a headline (pass rate, deterministic-only pass rate, cost, tokens, turns,
wall-clock), per-category and per-task pass/fail, per-task cost deltas, and where the two agents'
tool sequences forked. Two honesty rules are enforced in code: reports built on different suite
versions are refused rather than averaged, and the deterministic-only pass rate always sits next
to the headline one. See [`README.md`](README.md) for how to read the output.

## 6. Confirm the numbers are stable

A number you can't reproduce isn't evidence. The reproducibility harness runs the suite several
times and checks the pass set is stable:

```bash
uv run --package evaluation evaluation repro --track native --runs 3
```

It writes `docs/comparison/reproducibility.md`. An unstable pass set must be fixed before any
number is quoted — temperature is 0 and fixtures are pinned precisely so this holds.

## Optional — persist runs to Postgres

Alongside the JSON records, a run can be written to the evaluation tables
(`evaluation_runs`, `evaluation_results` — see [`../architecture/database-design.mermaid`](../architecture/database-design.mermaid)):

```bash
uv sync --package evaluation --extra native --extra postgres
uv run --package evaluation evaluation run --track native --postgres "postgresql://user:pass@host/db"
```

The tables are created on first use. Both the JSON record and the row are keyed by the same run
id, so a later `compare` finds either.

## Troubleshooting

- **`No Groq API key found`** — set `GROQ_API_KEY`, add a `.env` at the repo root, or use
  `--provider ollama`.
- **`evaluation: command not found`** — run it through uv: `uv run --package evaluation evaluation list`.
- **A `compare` that refuses** — the run records are on different suite versions; re-run the
  tracks against the same suite so they are comparable.
- **Only a native column appears** — expected until the langgraph track is built; `compare`
  renders whatever tracks it finds in `results/`.

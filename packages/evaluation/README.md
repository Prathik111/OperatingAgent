# evaluation

Reusable evaluation helpers for OperatingAgent.

This package provides:

- suite definitions for benchmark cases
- a generic runner that executes a case through an injected track callback
- a comparison renderer for side-by-side reporting

## Usage

```bash
uv run --package evaluation evaluation suite --out evaluation-suite.json
uv run --package evaluation evaluation compare native-results.json langgraph-results.json
```

The runner is intentionally callback-driven so it can be wired to the native or
LangGraph track without hard-coding orchestration details into the harness.
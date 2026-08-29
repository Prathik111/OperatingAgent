# agent-native v0.3.0 — backup

This is the original `agent-native` package as it stood on 2026-08-19, moved here
untouched before the v2 rebuild began.

It is the plan-and-execute + ReAct implementation reviewed in
`docs/review/native-agent-harness-review.md`: `agent.py` (step machinery),
`planner/`, `executor/`, `reflector/`, `verifier/`, `risk/`, `approval/`,
`compactor/`, `mcp/`, `sandbox/`, `tracing/`, `repository/`, `config.py`,
`harness/`, plus its 12 test files (86 tests).

It is kept for reference and for the "documented alternative orchestration"
option in the thesis. Nothing in the live package imports from here.

The target design it is being replaced by is in `docs/architecture/native-agent-v2.md`
and the four `docs/architecture/native-agent-v2-*.mermaid` diagrams.

This folder is deliberately **not** a workspace member: the root workspace globs
`packages/*` (one level), so `packages/agent-native/backup/...` is invisible to
`uv`, and the live package's `testpaths = ["tests"]` keeps pytest out of it.

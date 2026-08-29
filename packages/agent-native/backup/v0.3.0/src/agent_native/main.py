"""agent-native CLI - run a goal through the native agent.

    uv run --package agent-native python -m agent_native "goal text" [options]
    # or: uv run --package agent-native agent-native "goal text"

Decisions surfaced here:
  #4  --approval-timeout-s (default 120) - timeout auto-denies.
  #9  --db postgres|memory (default: postgres, falls back to memory with a
      printed notice when unreachable).
  #10 CLI is the entry point packages/api can adapt to later; api untouched.

Events are streamed to stdout as `[kind] payload` lines; approvals have no
interactive resolver by default (--approve-all / --approve-deny short-circuit
them for demos/CI).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import warnings

from .agent import build_agent, new_task
from .approval import ApprovalGateway
from .config import load_settings
from .events import APPROVAL_REQUESTED, AgentEvent
from .types import ApprovalDecision


class _FixedResolver(ApprovalGateway):
    def __init__(self, decision: ApprovalDecision, **kwargs) -> None:
        super().__init__(**kwargs)
        self._decision = decision

    async def request_approval(self, task_id: str, step) -> ApprovalDecision:
        return self._decision


def _print_event(event: AgentEvent) -> None:
    line = f"[{event.kind}]"
    if event.payload:
        line += " " + json.dumps(event.payload, ensure_ascii=False)
    print(line, flush=True)


async def _resolve_approvals(stdin_mode: bool, approve_all: bool, approve_deny: bool) -> ApprovalGateway | None:
    if approve_all:
        return _FixedResolver(ApprovalDecision.APPROVED, timeout_s=120.0)
    if approve_deny:
        return _FixedResolver(ApprovalDecision.DENIED, timeout_s=120.0)
    return ApprovalGateway(timeout_s=120.0, on_event=_print_event)


async def _run(goal: str, args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.llm_provider = args.provider or settings.llm_provider
    if args.approval_timeout_s:
        settings.approval_timeout_s = args.approval_timeout_s
    if args.no_sandbox:
        settings.sandbox.enabled = False

    from .repository import InMemoryTaskRepository, PostgresTaskRepository

    repository: object = InMemoryTaskRepository()
    if args.db == "postgres":
        try:
            repository = PostgresTaskRepository(settings.database_url)
            await repository._connect()  # noqa: SLF001 - eager check for fallback notice
            print(f"[db] postgres @ {settings.database_url}", flush=True)
        except Exception as e:
            print(f"[db] postgres unavailable ({e}); using in-memory repository", flush=True)
            repository = InMemoryTaskRepository()

    approval = await _resolve_approvals(
        stdin_mode=args.approve_stdin, approve_all=args.approve_all,
        approve_deny=args.approve_deny,
    )
    agent = build_agent(settings, repository=repository, approval=approval, on_event=_print_event)
    event_types = set()

    async def collect(event: AgentEvent) -> None:
        event_types.add(event.kind)
        _print_event(event)
        if event.kind == APPROVAL_REQUESTED and args.approve_stdin:
            step_id = event.payload.get("step_id")
            if step_id:
                answer = input(f"approve step {step_id}? (y/n) [n] ").strip().lower()
                decision = ApprovalDecision.APPROVED if answer == "y" else ApprovalDecision.DENIED
                approval.resolve(step_id, decision)

    task = new_task(goal=goal)
    result = await agent.run(task, on_event=collect)
    print(json.dumps({
        "status": result.status.value,
        "output": result.output,
        "llm_calls": result.llm_calls,
        "tool_calls": result.tool_calls,
        "total_tokens": result.total_tokens,
        "replans": result.replans,
        "failure_reason": result.failure_reason,
        "duration_ms": round(result.duration_ms, 1),
        "events": sorted(event_types),
    }, ensure_ascii=False), flush=True)
    if repository is not None and hasattr(repository, "close"):
        await repository.close()
    return 0 if result.status.value == "completed" else 1


def main() -> None:
    # Windows proactor: subprocess pipe transports warn at interpreter exit
    # after asyncio.run closes the loop (ValueError inside __del__/repr).
    # Cosmetic; the run has already finished by then.
    warnings.filterwarnings("ignore", category=ResourceWarning, module="asyncio")
    parser = argparse.ArgumentParser(prog="agent-native", description=__doc__)
    parser.add_argument("goal", help="goal for the agent to execute")
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None)
    parser.add_argument("--db", choices=["postgres", "memory"], default="postgres")
    parser.add_argument("--approval-timeout-s", type=float, default=None)
    parser.add_argument("--no-sandbox", action="store_true")
    parser.add_argument("--approve-all", action="store_true", help="auto-approve REVIEW calls")
    parser.add_argument("--approve-deny", action="store_true", help="auto-deny REVIEW calls")
    parser.add_argument("--approve-stdin", action="store_true", help="prompt on stdin for approvals")
    args = parser.parse_args()

    if not args.goal:
        parser.error("a goal is required")
    sys.exit(asyncio.run(_run(args.goal, args)))


if __name__ == "__main__":
    main()
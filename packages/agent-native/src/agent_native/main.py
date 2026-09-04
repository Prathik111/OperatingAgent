"""A thin command-line way to talk to the agent.

It thinks with a real model - Groq in the cloud (the default) or a model running
on your own machine through Ollama. For Groq, set `GROQ_API_KEY` in your
environment or a `.env` file at the repo root and run:

    uv run agent-native -m "read config.txt and tell me the port" --dir .

For a local model instead, install the extra (`uv sync --all-packages --extra ollama`), start
Ollama, pull a model, and run:

    uv run agent-native -m "..." --provider ollama --model qwen3.5:4b-q4_K_M

The agent's tools come from the MCP gateway (filesystem, git, terminal, search),
reached in-memory in this same process. That needs the MCP extra installed
(`uv sync --all-packages`). The filesystem tools are confined to `--dir`, and the
paths you ask about are read relative to it, so `--dir` must be a real folder.

Anything that only reads is allowed straight through. Anything that can change
files, delete, or run a shell command stops and asks you on the terminal first -
even by default. Pass `--yes` to approve those without prompting (handy for an
unattended run, but it means the agent can write and run commands on its own).

Shell commands run inside a container when Docker is available: only `--dir` is
mounted, there's no network, and memory and CPU are capped, so a bad command can't
reach the rest of your machine. Without Docker they run on your machine instead,
limited to a short allowlist of inspection commands. The first line of every run
says which of the two you got; `--sandbox on` refuses to run without a container,
and `--sandbox off` skips it.

Every run ends with a one-line receipt (tokens, cost, time) and leaves a timing
trace at `.agent-traces/<run id>.json` inside `--dir`, which is where to look
when a run felt slow. `--no-traces` turns the file off.

A run can be handed a budget. `--max-cost 0.05` stops it once it has spent five
cents on the chosen model, and `--max-tokens 200000` once input and output tokens
reach that many. Either way it stops cleanly between turns and hands back what it
had, with the ceiling it hit named on the receipt. Both are off by default.

By default nothing is kept once the process ends. Pass `--database sqlite` or
`--database postgres://...`
to keep sessions, messages, events and run receipts.

Once receipts are kept, `agent-native runs --database sqlite` or
`--database postgres://...` lists recent
runs - turns, tokens, cost and time, with a totals line - for a session
(`--session`), a folder (`--dir`), or every session. It only reads, so it's the
place to answer "did it get cheaper?" without re-running anything.

`agent-native sessions --database sqlite` or `--database postgres://...` manages
those stored sessions:
`sessions list` shows them newest-first, each with its last receipt; `sessions
fork <id>` branches a conversation into a new session to try an alternative from
the same history; `sessions delete <id>` removes one and everything under it.
None of these need a model, so no key is asked for; resuming an interrupted run
does, so that stays on the run path (and the HTTP API's resume route).

Two things carry over between conversations. Write an `AGENT.md` in `--dir` and its
contents are added to the agent's instructions every session - that's the place for
"always run the tests with uv". The agent can also keep short notes of its own with
the remember tool (it asks first, and shows you the note), and those come back at
the start of later sessions. Notes need `--database`, since anything in memory is
gone when the process is.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from sandbox import DEFAULT_IMAGE, ContainerSandbox

from .checkpoint import CheckpointStore, default_base_for, install_auto_checkpoint
from .config import AgentConfig
from .events import EventType
from .loop import Limits
from .models.base import Model, ToolFormat
from .models.groq_model import GROQ_MODELS, Groq
from .models.ollama_model import Ollama
from .monitoring import Monitoring
from .permissions import (
    PermissionAnswer,
    PermissionDuration,
    PermissionRequest,
    PermissionResponder,
)
from .service import AgentRuntime, AgentService
from .tools.mcp_bridge import MCPToolProvider
from .tools.subagent import is_helper_run

# A sensible default model for each backend, used when --model is omitted.
DEFAULT_MODELS = {
    # This model is available to the project's configured Groq account and
    # supports the structured/tool responses used by the native loop.
    "groq": "gpt-oss-20b",
    "ollama": "qwen3.5:4b-q4_K_M",
}

#: Where run traces go when --trace-dir isn't given. Inside the working folder,
#: dot-prefixed, so it sits next to the work it describes and stays out of the way.
DEFAULT_TRACE_DIR = ".agent-traces"


def _load_environment() -> None:
    """Load the nearest ``.env`` for standalone CLI runs.

    The API entrypoint already does this before startup. The native CLI is also a
    supported entrypoint, so it must load the same provider and Langfuse settings
    before constructing ``Monitoring`` or a model provider. Exported variables
    remain authoritative.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def _build_runtime(args: argparse.Namespace, workdir: Path, database=None) -> AgentRuntime:
    """Wire a runtime around the chosen backend (Groq in the cloud, or Ollama locally)."""
    model_name = args.model or DEFAULT_MODELS[args.provider]
    config = AgentConfig(
        name=args.agent,
        model=model_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
    )
    trace_dir = None if args.no_traces else Path(args.trace_dir or (workdir / DEFAULT_TRACE_DIR))
    runtime = AgentRuntime(
        database=database,
        agents=[config],
        monitoring=Monitoring(trace_dir=trace_dir, otlp_endpoint=args.otlp_endpoint or ""),
        sandbox=_build_sandbox(args),
    )

    if args.provider == "ollama":
        _wire_ollama(runtime, model_name, args.ollama_host)
    else:
        _wire_groq(runtime, model_name)

    return runtime


def _build_sandbox(args: argparse.Namespace):
    """The container shell commands run in, unless it's switched off.

    `auto` (the default) builds one and lets it fall back on a machine without
    Docker; `off` returns None so commands run on the machine as before; `on` is
    checked at startup and refuses to run rather than silently falling back, which
    is what you want on a shared machine or in CI.
    """
    if args.sandbox == "off":
        return None

    return ContainerSandbox(
        image=args.sandbox_image or DEFAULT_IMAGE,
        network=args.sandbox_network,
    )


async def _open_database(dsn: str):
    """Open the store the user asked for. `memory` (the default) means None.

    Returning None lets `AgentRuntime` build its own `MemoryDatabase`, so the
    default path has no database code in it at all. ``sqlite`` uses the standard
    per-user application data directory; ``sqlite:///path`` selects an explicit
    file.
    """
    if not dsn or dsn == "memory":
        return None
    if dsn.lower() == "sqlite" or dsn.lower().startswith("sqlite:"):
        from .sqlite import SQLiteDatabase

        if dsn.lower() == "sqlite":
            if os.name == "nt":
                root = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
                base = (
                    Path(root) / "OperatingAgent"
                    if root
                    else Path.home() / "OperatingAgent"
                )
            elif sys.platform == "darwin":
                base = Path.home() / "Library" / "Application Support" / "OperatingAgent"
            else:
                base = Path(
                    os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share"
                ) / "OperatingAgent"
            path = base / "operating-agent.db"
        elif dsn.lower().startswith("sqlite:///"):
            raw = dsn[len("sqlite:///") :]
            path = Path(
                raw
                if os.name == "nt" and len(raw) > 1 and raw[1] == ":"
                else f"/{raw}"
            )
        else:
            path = Path(dsn[len("sqlite:") :].lstrip("/"))
        return SQLiteDatabase(path)
    from .postgres import PostgresDatabase

    return await PostgresDatabase.open(dsn)


def _wire_groq(runtime: AgentRuntime, model_name: str) -> None:
    """Register the Groq provider and the model to think with."""
    groq = Groq()
    if not groq.has_key:
        raise SystemExit(
            "No Groq API key found.\n"
            "  Set it in your shell:  export GROQ_API_KEY=gsk_...\n"
            "  Or put this line in a .env file at the repo root:\n"
            "      GROQ_API_KEY=gsk_...\n"
            "  Or run against a local model instead:  --provider ollama\n"
            "Then run the same command again."
        )

    runtime.models.register_provider("groq", groq)
    # A known short name (llama-3.3-70b) maps to a full Groq model id and window;
    # anything else is passed straight through as a Groq model id.
    model = GROQ_MODELS.get(model_name) or Model(
        provider="groq", model_id=model_name, context_size=128_000, max_output=8192
    )
    runtime.models.register_model(model_name, model)


def _wire_ollama(runtime: AgentRuntime, model_name: str, host: str) -> None:
    """Register the Ollama provider and the local model to think with.

    No API key needed - it just needs the Ollama app running and the model
    pulled (`ollama pull <name>`). The window is kept modest so a small local
    model loads comfortably; pass a bigger --model or edit here to go wider.
    """
    runtime.models.register_provider("ollama", Ollama(host=host))
    model = Model(
        provider="ollama",
        model_id=model_name,
        context_size=8192,
        max_output=2048,
        tool_format=ToolFormat.NATIVE,
    )
    runtime.models.register_model(model_name, model)


async def _run(args: argparse.Namespace) -> int:
    _load_environment()
    # Resolve the working folder first. This is the one folder the agent's file
    # tools may touch, so it has to exist before we start - a clear message here
    # beats a confusing "file not found" on every path the model tries. It also
    # decides where run traces are written, so it's needed before the runtime.
    workdir = Path(args.dir).expanduser().resolve()
    if not workdir.is_dir():
        print(
            f"[error] --dir isn't an existing folder: {workdir}\n"
            "        Point --dir at a real folder; the agent's file tools are "
            "confined to it.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    try:
        database = await _open_database(args.database)
    except (RuntimeError, OSError, ImportError) as exc:
        print(
            f"[error] couldn't open the database: {exc}\n"
            "        Drop --database to run against memory instead.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    runtime = _build_runtime(args, workdir, database)

    # Optional safety net: with --checkpoint, snapshot the folder before the run's
    # first edit so a wrong or destructive change can be rewound afterwards. Off
    # unless asked for - installing nothing leaves the run byte-for-byte as it was,
    # the hook layer's standing promise. The store is kept outside the folder and
    # keyed to it, so `checkpoints rewind` in a later process finds it.
    checkpoint_store = (
        install_auto_checkpoint(runtime, base_dir=default_base_for(str(workdir)))
        if getattr(args, "checkpoint", False)
        else None
    )

    # Say which mode this run is in before it starts, not after. A user who thinks
    # commands are contained when they aren't is the one bad outcome here.
    if runtime.sandbox is not None:
        ready = await runtime.sandbox.probe()
        print(f"[{runtime.sandbox.status_line()}]", flush=True)
        if args.sandbox == "on" and not ready:
            print(
                "[error] --sandbox on was asked for and no container is available.\n"
                "        Start Docker, or use --sandbox auto to fall back to the "
                "machine, or --sandbox off to say so plainly.",
                file=sys.stderr,
                flush=True,
            )
            await runtime.sandbox.close()
            await runtime.database.close()
            return 1
    else:
        print("[sandbox: off - shell commands run on your machine]", flush=True)

    # Borrow the whole tool fleet from the MCP gateway, in-memory. This must
    # happen before the session is created, so the system prompt lists the tools.
    # Passing the working folder as the root confines the file tools to it.
    mcp = MCPToolProvider()
    try:
        tools = await mcp.connect(root=str(workdir))
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        await mcp.close()
        if runtime.sandbox is not None:
            await runtime.sandbox.close()
        await runtime.database.close()
        return 1
    for tool in tools:
        runtime.tools.register(tool)

    service = AgentService(runtime)
    session = await service.create_session(agent=args.agent, working_directory=str(workdir))

    # Where approvals come from. --yes approves risky calls without asking (handy
    # unattended, but the agent can then write and run commands on its own); without
    # it, each risky call stops and asks on the terminal. Only risky calls ever reach
    # a responder - reads are allowed by policy and never ask.
    runtime.permissions.set_responder(AutoApproveResponder() if args.yes else TerminalResponder())

    async def print_events() -> None:
        async for event in service.subscribe(session.id):
            if event.type == EventType.ASSISTANT_DELTA:
                print(event.data.get("text", ""), end="", flush=True)
            elif event.type == EventType.TOOL_STARTED:
                print(f"\n[tool] {event.data.get('name')}: {event.data.get('preview', '')}", flush=True)
            elif event.type == EventType.TOOL_FINISHED:
                ok = "ok" if event.data.get("success") else "failed"
                detail = event.data.get("error") or ""
                print(f"\n[tool] {event.data.get('name')} {ok}{': ' + detail if detail else ''}", flush=True)
            elif event.type == EventType.ERROR:
                print(f"\n[error] {event.data.get('error')}", file=sys.stderr, flush=True)
            elif event.type == EventType.RUN_FINISHED:
                # A helper's run finishes on this same stream. Print its receipt,
                # but don't stop watching - the conversation isn't over.
                if is_helper_run(event.data.get("run_id", "")):
                    print("\n" + _receipt(event.data, helper=True), flush=True)
                    continue
                print("\n\n" + _receipt(event.data), flush=True)
                return

    printer = asyncio.create_task(print_events())

    print(f"> {args.message}\n")
    try:
        result = await service.send_message(
            session.id,
            args.message,
            limits=Limits(
                max_turns=args.max_turns,
                max_cost_usd=args.max_cost,
                max_total_tokens=args.max_tokens,
                plan_mode=args.plan_mode,
            ),
        )
        await printer
        if result.trace_id:
            from observability import get_trace_url

            trace_url = get_trace_url(result.trace_id)
            if trace_url:
                print(f"[trace] {trace_url}", flush=True)
    finally:
        if not printer.done():
            printer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await printer
        await mcp.close()
        # The container goes before the event bus and the database: it's the only
        # thing here that outlives the process if we don't clean it up.
        if runtime.sandbox is not None:
            await runtime.sandbox.close()
        await runtime.events.close()
        await runtime.database.close()
        for path in runtime.monitoring.shutdown():
            print(f"[trace] {path}", flush=True)
        # Say what happened to the collector, but only when one was asked for: a
        # confirmed export, or the reason it didn't (extra missing, collector down).
        mon = runtime.monitoring
        if mon.otlp_attempted:
            if mon.otlp_exported:
                print("[trace] spans exported to the OTLP collector", flush=True)
            else:
                print(f"[trace] OTLP export skipped: {mon.otlp_skipped_reason}", flush=True)

    # If the safety net caught a pre-edit snapshot, say so and how to undo it.
    if checkpoint_store is not None and checkpoint_store.list():
        cp = checkpoint_store.latest()
        assert cp is not None
        print(
            f"[checkpoint] snapshot of the folder taken before edits "
            f"({cp.file_count} files). Undo them with: "
            f'agent-native checkpoints rewind --dir "{workdir}"',
            flush=True,
        )

    # A run that hit its turn limit still did real work, so it isn't a failure.
    return 0 if result.status.value in ("finished", "limit_reached") else 1


def _receipt(data: dict, helper: bool = False) -> str:
    """One line saying how the run went, what it used, and what it cost.

    Cost is shown to six decimals because a single cheap run really does cost
    fractions of a cent, and rounding that to two would print $0.00 and teach the
    user nothing. Retries only appear when there were some. A helper's line is
    labelled with its run id so its cost is visibly separate from the parent's. A
    run stopped by a budget or turn/time ceiling adds `reason=...`, so the receipt
    says which limit ended it, not just that one did. When a reasoning model breaks
    out its thinking, `(N reasoning)` sits beside the output count - it's part of
    that `out` number, not on top of it (see Usage), so cost is unchanged; it just
    makes a thinking budget's effect visible. Ollama, which doesn't report it,
    shows nothing there.
    """
    cached = data.get("cached_tokens") or 0
    reasoning = data.get("reasoning_tokens") or 0
    parts = [
        f"status={data.get('status')}",
        f"turns={data.get('turns')}",
        f"time={data.get('duration_seconds', 0.0)}s",
        f"tokens={data.get('input_tokens', 0)} in / {data.get('output_tokens', 0)} out"
        + (f" ({cached} cached)" if cached else "")
        + (f" ({reasoning} reasoning)" if reasoning else ""),
        f"cost=${data.get('cost_usd', 0.0):.6f}",
    ]
    if data.get("stop_reason"):
        parts.append(f"reason={data['stop_reason']}")
    if data.get("retries"):
        parts.append(f"retries={data['retries']}")
    if data.get("model"):
        parts.append(f"model={data['model']}")
    if data.get("trace_id"):
        parts.append(f"trace={data['trace_id']}")
    label = f"[helper {data.get('run_id', '')}]" if helper else "[done]"
    return label + " " + " ".join(parts)


async def _ask_on_terminal(request) -> tuple:
    """Ask the user to approve one risky call: (allowed, for how long, where).

    Two answers aren't enough. "Yes" every single time is how a user learns to
    stop reading the prompt, and "yes to everything for the session" is more than
    most people mean - so there's a third answer in between: yes for the session,
    but only under this folder. Typing `s notes` approves writes under `notes/`
    and keeps the prompt for anything else.

    No terminal means no. A run in a pipe or a cron job cannot be asked, and the
    safe reading of silence on a destructive call is refusal.
    """
    prompt = (
        f"\n[permission needed] {request.preview}"
        f"\n  why: {request.reason}"
        f"\n  allow? y = once, n = no, s = rest of session"
        f'\n         (add a folder to limit it, e.g. "s notes") [y/N/s] '
    )
    try:
        answer = await asyncio.get_running_loop().run_in_executor(None, input, prompt)
    except (EOFError, OSError, RuntimeError):
        print("  no terminal to ask on; denying to be safe.", flush=True)
        return False, PermissionDuration.ONCE, ""
    return _read_answer(answer)


def _read_answer(answer: str) -> tuple:
    """Turn what was typed into (allowed, duration, scope). Anything unclear is no."""
    words = (answer or "").strip().split()
    if not words:
        return False, PermissionDuration.ONCE, ""
    head = words[0].lower()
    scope = " ".join(words[1:]).strip().strip("\"'")
    if head in ("s", "session"):
        return True, PermissionDuration.SESSION, scope
    if head in ("y", "yes"):
        return True, PermissionDuration.ONCE, ""
    return False, PermissionDuration.ONCE, ""


class TerminalResponder(PermissionResponder):
    """Answers approvals inline, by prompting on stdin.

    The CLI's implementation of the `PermissionResponder` seam: when the manager
    needs a decision, this asks the person at the terminal and hands the answer
    straight back - no out-of-band delivery, so `deliver`/`pending` stay the no-ops
    the base class defines. The prompt and its grammar (`y` / `n` / `s notes`) are
    `_ask_on_terminal` and `_read_answer`, unchanged; this only adapts their tuple
    to a `PermissionAnswer`.
    """

    async def respond(self, request: PermissionRequest) -> PermissionAnswer:
        allowed, duration, scope = await _ask_on_terminal(request)
        return PermissionAnswer(allowed, duration, scope)


class AutoApproveResponder(PermissionResponder):
    """Approves every request without asking - the `--yes` path.

    Says the same "approving" line the old polling loop did, so an unattended run
    still shows what it waved through, then returns a one-off yes (it never widens
    to a session grant on the user's behalf).
    """

    async def respond(self, request: PermissionRequest) -> PermissionAnswer:
        print(f"\n[permission] approving (--yes): {request.preview}", flush=True)
        return PermissionAnswer(True, PermissionDuration.ONCE, "")


# ---------------------------------------------------------------------------
# `agent-native runs`: the run-history view
# ---------------------------------------------------------------------------
async def _runs_view(argv: list) -> int:
    """List recent runs with their receipts - for a session, a folder, or all.

    The numbers are already stored: every run writes a receipt through `save_run`.
    This is the surface that reads them back, so "did it get cheaper?" is a query
    rather than a scroll through old terminal output. It wants a durable store -
    `--database postgres://...` - since a fresh process's memory store is empty. The
    view is read-only: it opens the store, lists, prints, and closes.
    """
    parser = argparse.ArgumentParser(
        prog="agent-native runs",
        description="List recent runs with their receipts (turns, tokens, cost, time).",
    )
    parser.add_argument(
        "--database",
        default="memory",
        help=(
            "Where to read runs from. `memory` (the default) is empty in a fresh "
            "process, so this view wants `sqlite` or a Postgres URL like "
            "postgres://user:pass@localhost/agent - the same store the run used."
        ),
    )
    parser.add_argument(
        "--session",
        default="",
        help="Only runs from this session id. Takes precedence over --dir.",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Only runs from sessions opened in this folder (resolved like a run's --dir).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many recent runs to show (0 = all). Default 20.",
    )
    args = parser.parse_args(argv)

    try:
        database = await _open_database(args.database)
    except (RuntimeError, OSError, ImportError) as exc:
        print(f"[error] couldn't open the database: {exc}", file=sys.stderr, flush=True)
        return 1

    if database is None:
        # The memory store lives and dies with its process, so a second process has
        # nothing to show. Say so plainly rather than printing an empty table.
        print(
            "No run history in the memory store - it's emptied when the process ends.\n"
            "Re-run with --database postgres://... (the same store the run used) to "
            "see past runs.",
            flush=True,
        )
        return 0

    try:
        runs = await _list_runs_for_view(database, args)
    finally:
        await database.close()

    if not runs:
        if args.session:
            where = f" for session {args.session}"
        elif args.dir:
            where = f" under {Path(args.dir).expanduser().resolve()}"
        else:
            where = ""
        print(f"No runs found{where}.", flush=True)
        return 0

    print(_render_runs_table(runs), flush=True)
    return 0


async def _list_runs_for_view(database, args: argparse.Namespace) -> list:
    """Pick the runs the view asked for: one session, one folder, or everything.

    `--session` is the narrowest and wins outright. `--dir` resolves to a folder,
    finds the sessions opened there, and keeps the runs that belong to them -
    reusing `list_runs`' own newest-first ordering by filtering it, so no run needs
    to carry a timestamp for the merge. With neither, it's the most recent runs
    across every session.
    """
    limit = args.limit if args.limit and args.limit > 0 else 0
    if args.session:
        return await database.list_runs(session_id=args.session, limit=limit)
    if args.dir:
        workdir = str(Path(args.dir).expanduser().resolve())
        sessions = await database.list_sessions(working_directory=workdir)
        ids = {s.id for s in sessions}
        # list_runs("") is already globally newest-first; filtering by membership
        # keeps that order, then the cap applies to the folder's own runs.
        folder_runs = [
            r for r in await database.list_runs("", 0) if getattr(r, "session_id", "") in ids
        ]
        return folder_runs[:limit] if limit else folder_runs
    return await database.list_runs("", limit)


def _runs_totals(runs: list) -> dict:
    """Sum the receipt columns across the listed runs.

    Pure, so a test can check the totals against the rows without going through the
    printed table - which is exactly the plan's verify for this step.
    """
    totals = {
        "runs": len(runs),
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
    }
    for r in runs:
        totals["turns"] += int(getattr(r, "turns", 0) or 0)
        totals["input_tokens"] += int(getattr(r, "input_tokens", 0) or 0)
        totals["output_tokens"] += int(getattr(r, "output_tokens", 0) or 0)
        totals["cached_tokens"] += int(getattr(r, "cached_tokens", 0) or 0)
        totals["reasoning_tokens"] += int(getattr(r, "reasoning_tokens", 0) or 0)
        totals["cost_usd"] += float(getattr(r, "cost_usd", 0.0) or 0.0)
        totals["duration_seconds"] += float(getattr(r, "duration_seconds", 0.0) or 0.0)
    return totals


def _render_runs_table(runs: list) -> str:
    """The runs as an aligned text table, with a TOTALS line beneath it.

    Columns are the same receipt a run prints when it finishes, lined up so each
    reads down the page: run id, status, turns, tokens in/out (and cached, and
    reasoning, each shown only when some run had any), cost, wall-clock, retries
    (when any) and model. The TOTALS row sums the numeric columns - the point of
    the step, turning "did it get cheaper" into a number you can read at a glance.
    Cost keeps six decimals for the same reason the receipt does: a cheap run
    rounds to $0.00 and teaches nothing.
    """
    show_cached = any(int(getattr(r, "cached_tokens", 0) or 0) for r in runs)
    show_reasoning = any(int(getattr(r, "reasoning_tokens", 0) or 0) for r in runs)
    show_retries = any(int(getattr(r, "retries", 0) or 0) for r in runs)

    headers = ["RUN", "STATUS", "TURNS", "IN", "OUT"]
    aligns = ["l", "l", "r", "r", "r"]
    if show_cached:
        headers.append("CACHED")
        aligns.append("r")
    if show_reasoning:
        headers.append("REASONING")
        aligns.append("r")
    headers += ["COST", "TIME"]
    aligns += ["r", "r"]
    if show_retries:
        headers.append("RETRIES")
        aligns.append("r")
    headers.append("MODEL")
    aligns.append("l")

    def row_for(r: Any) -> list:
        cells = [
            str(getattr(r, "run_id", "") or ""),
            str(getattr(r, "status", "") or ""),
            str(int(getattr(r, "turns", 0) or 0)),
            str(int(getattr(r, "input_tokens", 0) or 0)),
            str(int(getattr(r, "output_tokens", 0) or 0)),
        ]
        if show_cached:
            cells.append(str(int(getattr(r, "cached_tokens", 0) or 0)))
        if show_reasoning:
            cells.append(str(int(getattr(r, "reasoning_tokens", 0) or 0)))
        cells += [
            f"{float(getattr(r, 'cost_usd', 0.0) or 0.0):.6f}",
            f"{float(getattr(r, 'duration_seconds', 0.0) or 0.0):.2f}s",
        ]
        if show_retries:
            cells.append(str(int(getattr(r, "retries", 0) or 0)))
        cells.append(str(getattr(r, "model", "") or ""))
        return cells

    rows = [row_for(r) for r in runs]

    totals = _runs_totals(runs)
    footer = [
        f"TOTALS ({totals['runs']} run{'s' if totals['runs'] != 1 else ''})",
        "",
        str(totals["turns"]),
        str(totals["input_tokens"]),
        str(totals["output_tokens"]),
    ]
    if show_cached:
        footer.append(str(totals["cached_tokens"]))
    if show_reasoning:
        footer.append(str(totals["reasoning_tokens"]))
    footer += [f"{totals['cost_usd']:.6f}", f"{totals['duration_seconds']:.2f}s"]
    if show_retries:
        footer.append("")  # a sum of retries across runs isn't a meaningful receipt line
    footer.append("")

    return _format_table(headers, rows, aligns, footer=footer)


def _format_table(
    headers: list, rows: list, aligns: list, footer: list | None = None
) -> str:
    """A minimal fixed-width table: header, a rule, the rows, and an optional footer.

    Each column is as wide as its widest cell (header and footer counted), so the
    columns line up whatever the data holds. `aligns` is 'l' or 'r' per column.
    """
    body = [headers, *rows, *([footer] if footer else [])]
    widths = [max(len(str(row[i])) for row in body) for i in range(len(headers))]

    def fmt(row: list) -> str:
        out = []
        for i, cell in enumerate(row):
            text = str(cell)
            out.append(text.rjust(widths[i]) if aligns[i] == "r" else text.ljust(widths[i]))
        return "  ".join(out).rstrip()

    rule = "  ".join("-" * w for w in widths)
    lines = [fmt(headers), rule, *[fmt(r) for r in rows]]
    if footer:
        lines += [rule, fmt(footer)]
    return "\n".join(lines)


def _checkpoints_view(argv: list) -> int:
    """`agent-native checkpoints <list|rewind> --dir <folder>` - the edit safety net.

    A run started with --checkpoint snapshots the working folder before its first
    edit. This view reopens that folder's snapshot store (kept outside the folder,
    keyed to its path) so you can see the snapshots or rewind the folder to one -
    the byte-for-byte undo for a change you didn't want. Filesystem-only: it opens
    no database, loads no model, and needs no key.
    """
    parser = argparse.ArgumentParser(
        prog="agent-native checkpoints",
        description="List or rewind filesystem checkpoints of a working folder.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dir_help = "The working folder whose checkpoints to act on (the run's --dir)."
    p_list = sub.add_parser("list", help="List snapshots taken for a folder, newest first.")
    p_list.add_argument("--dir", default=".", help=dir_help)

    p_rewind = sub.add_parser(
        "rewind", help="Restore the folder to a checkpoint, byte for byte (newest by default)."
    )
    p_rewind.add_argument("--dir", default=".", help=dir_help)
    p_rewind.add_argument(
        "id", nargs="?", default="", help="Which checkpoint to restore. Omit for the most recent."
    )

    args = parser.parse_args(argv)

    workdir = Path(args.dir).expanduser().resolve()
    store = CheckpointStore.load(default_base_for(str(workdir)))
    snapshots = store.list()

    if args.command == "list":
        if not snapshots:
            print(
                f"No checkpoints for {workdir}.\n"
                "Run with --checkpoint to snapshot the folder before edits.",
                flush=True,
            )
            return 0
        print(_render_checkpoints_table(snapshots), flush=True)
        return 0

    if args.command == "rewind":
        if not snapshots:
            print(f"No checkpoints for {workdir} to rewind to.", file=sys.stderr, flush=True)
            return 1
        target = store.get(args.id) if args.id else store.latest()
        if target is None:
            print(f"No such checkpoint: {args.id}", file=sys.stderr, flush=True)
            return 1
        store.restore(target)
        print(f"Rewound {workdir} to {target.id} ({target.file_count} files).", flush=True)
        return 0

    return 0  # argparse's required subparser makes this unreachable


def _render_checkpoints_table(snapshots: list) -> str:
    """Checkpoints as an aligned table: id, when it was taken, file count, label."""
    import datetime as _dt

    rows = [("ID", "TAKEN", "FILES", "LABEL")]
    for cp in snapshots:
        when = (
            _dt.datetime.fromtimestamp(cp.created_at, _dt.UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if cp.created_at
            else "-"
        )
        rows.append((cp.id, when, str(cp.file_count), cp.label or "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    )


async def _sessions_view(argv: list) -> int:
    """`agent-native sessions <list|fork|delete> ...` - manage stored sessions.

    A companion to `runs`: list what's stored (each with its last receipt), branch
    one into a new session to try an alternative from the same history, or delete
    one and everything filed under it. Like `runs` it needs a real store - the
    memory store is empty in a fresh process - so every subcommand takes --database.

    None of these need a model. Listing, forking (which copies stored messages) and
    deleting are pure store operations, so this view builds a provider-free runtime
    and never asks for a Groq key. Resuming an interrupted run *does* need a model,
    so that stays on the run path and the API's POST /sessions/{id}/resume.
    """
    store_help = (
        "Where sessions are kept. `memory` (the default) is empty in a fresh "
        "process, so this wants `sqlite` or a Postgres URL like "
        "postgres://user:pass@localhost/agent - the same store the run used."
    )
    parser = argparse.ArgumentParser(
        prog="agent-native sessions", description="List, fork, or delete stored sessions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser(
        "list", help="List stored sessions, newest first, each with its last receipt."
    )
    p_list.add_argument("--database", default="memory", help=store_help)
    p_list.add_argument(
        "--dir", default="", help="Only sessions opened in this folder (resolved like a run's --dir)."
    )
    p_list.add_argument(
        "--limit", type=int, default=20, help="How many recent sessions to show (0 = all). Default 20."
    )

    p_fork = sub.add_parser("fork", help="Branch a session's conversation into a new session.")
    p_fork.add_argument("session", help="The id of the session to fork.")
    p_fork.add_argument("--database", default="memory", help=store_help)
    p_fork.add_argument(
        "--title", default="", help="Title for the new session (default: the source title with '(fork)')."
    )

    p_delete = sub.add_parser("delete", help="Delete a session and everything filed under it.")
    p_delete.add_argument("session", help="The id of the session to delete.")
    p_delete.add_argument("--database", default="memory", help=store_help)

    args = parser.parse_args(argv)

    try:
        database = await _open_database(args.database)
    except (RuntimeError, OSError, ImportError) as exc:
        print(f"[error] couldn't open the database: {exc}", file=sys.stderr, flush=True)
        return 1

    if database is None:
        # Same story as the runs view: the memory store is gone with its process, so
        # a second process has nothing to manage. Say so rather than act on nothing.
        print(
            "No stored sessions in the memory store - it's emptied when the process ends.\n"
            "Re-run with --database postgres://... (the same store the run used).",
            flush=True,
        )
        return 0

    try:
        return await _dispatch_sessions(args, database)
    finally:
        await database.close()


async def _dispatch_sessions(args: argparse.Namespace, database: Any) -> int:
    """Run the chosen `sessions` subcommand against an already-open store.

    Split out from `_sessions_view` so a test can drive the commands against a
    `MemoryDatabase` directly, without the argparse front door or a Postgres URL.
    Fork and delete go through `AgentService` on a provider-free runtime - the same
    code the API calls - so the two surfaces can't drift apart.
    """
    if args.command == "list":
        workdir = str(Path(args.dir).expanduser().resolve()) if args.dir else ""
        limit = args.limit if args.limit and args.limit > 0 else 0
        sessions = await database.list_sessions(working_directory=workdir, limit=limit)
        if not sessions:
            where = f" under {workdir}" if workdir else ""
            print(f"No sessions found{where}.", flush=True)
            return 0
        # Fold each session's most recent receipt onto its line, the same pairing the
        # API's session list returns; list_runs(id, 1) is that newest run or nothing.
        rows = [(s, ((await database.list_runs(s.id, 1)) or [None])[0]) for s in sessions]
        print(_render_sessions_table(rows), flush=True)
        return 0

    service = AgentService(AgentRuntime(database=database, agents=[AgentConfig()]))

    if args.command == "fork":
        try:
            fork = await service.fork_session(args.session, args.title)
        except KeyError:
            print(f"No such session: {args.session}", file=sys.stderr, flush=True)
            return 1
        print(f"Forked {args.session} -> {fork.id}", flush=True)
        return 0

    if args.command == "delete":
        existed = await service.delete_session(args.session)
        if not existed:
            print(f"No such session: {args.session}", file=sys.stderr, flush=True)
            return 1
        print(f"Deleted {args.session}", flush=True)
        return 0

    return 0  # argparse's required subparser makes this unreachable


def _render_sessions_table(rows: list) -> str:
    """Stored sessions as an aligned table: id, agent, title, folder, last receipt.

    `rows` is a list of ``(session, latest_run_or_None)`` - the same pairing the
    API's session list returns - so a test can check the rows without going through
    the printed table. The last-run columns fold a session's most recent receipt
    onto its line (status, cost, wall-clock), or a dash when it has no runs yet.
    """
    headers = ["SESSION", "AGENT", "TITLE", "DIR", "LAST RUN", "COST", "TIME"]
    aligns = ["l", "l", "l", "l", "l", "r", "r"]

    def row_for(pair: Any) -> list:
        session, run = pair
        if run is None:
            last, cost, when = "-", "", ""
        else:
            last = str(getattr(run, "status", "") or "")
            cost = f"{float(getattr(run, 'cost_usd', 0.0) or 0.0):.6f}"
            when = f"{float(getattr(run, 'duration_seconds', 0.0) or 0.0):.2f}s"
        return [
            str(getattr(session, "id", "") or ""),
            str(getattr(session, "agent", "") or ""),
            str(getattr(session, "title", "") or ""),
            str(getattr(session, "working_directory", "") or ""),
            last,
            cost,
            when,
        ]

    return _format_table(headers, [row_for(p) for p in rows], aligns)


def main() -> None:
    # A tiny front door before the main parser. `agent-native runs ...` is a
    # read-only history view and `agent-native sessions ...` manages stored
    # sessions - each with its own flags; everything else is a run, parsed exactly
    # as before. Dispatching on a bare first word (not a value the run parser could
    # ever produce, since that one always leads with -m/--message or another flag)
    # keeps the original `agent-native -m "..."` invocation untouched.
    argv = sys.argv[1:]
    if argv and argv[0] == "runs":
        raise SystemExit(asyncio.run(_runs_view(argv[1:])))
    if argv and argv[0] == "sessions":
        raise SystemExit(asyncio.run(_sessions_view(argv[1:])))
    if argv and argv[0] == "checkpoints":
        # Pure filesystem work - no store, no model - so it runs without asyncio.
        raise SystemExit(_checkpoints_view(argv[1:]))

    parser = argparse.ArgumentParser(prog="agent-native", description="Talk to the native agent.")
    parser.add_argument("--message", "-m", required=True, help="What to say to the agent.")
    parser.add_argument("--agent", default="build", help="Which agent to use.")
    parser.add_argument(
        "--dir",
        default=".",
        help="Working folder the agent's file tools are confined to (must exist).",
    )
    parser.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "ollama"],
        help="Where the model runs: groq (cloud, needs a key) or ollama (your machine).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model to think with. Omit to use the provider's default "
            "(groq: gpt-oss-20b; ollama: qwen3.5:4b-q4_K_M). For Groq, any "
            "Groq model id works; for Ollama, any pulled model name."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        dest="ollama_host",
        help="Ollama server URL (used with --provider ollama).",
    )
    parser.add_argument("--max-turns", type=int, default=10, dest="max_turns")
    parser.add_argument(
        "--max-cost",
        type=float,
        default=0.0,
        dest="max_cost",
        help=(
            "Stop the run once it has spent this many US dollars, priced on the "
            "chosen model. 0 (the default) means no cost ceiling. The run stops "
            "cleanly between turns and hands back what it had, with the reason on "
            "the receipt."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        dest="max_tokens",
        help=(
            "Stop the run once input+output tokens reach this many. 0 (the default) "
            "means no token ceiling. Stops cleanly between turns, like --max-cost."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        dest="plan_mode",
        help=(
            "Plan mode: the agent is shown only read-only tools and any mutating "
            "call is refused, so it investigates and proposes an approach without "
            "changing anything. Review the plan, then run again without --plan to "
            "let it act."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        dest="checkpoint",
        help=(
            "Snapshot the working folder before the run's first edit, so a wrong or "
            "destructive change can be undone. After the run, rewind with "
            "`agent-native checkpoints rewind --dir <folder>`. Snapshots are kept "
            "outside the folder, keyed to it, so a later rewind finds them."
        ),
    )
    parser.add_argument(
        "--database",
        default="memory",
        help=(
            "Where to keep sessions, messages, events and runs. `memory` (the "
            "default) forgets everything when the process ends; `sqlite` keeps a "
            "durable per-user file, `sqlite:///path` selects an explicit file, and "
            "a Postgres URL like postgres://user:pass@localhost/agent keeps them in "
            "Postgres. The Postgres extra must be installed: uv sync --all-packages "
            "--extra postgres"
        ),
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        dest="trace_dir",
        help=(
            "Where to write the run's timing trace (one JSON file per run). "
            f"Defaults to {DEFAULT_TRACE_DIR}/ inside --dir."
        ),
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        dest="no_traces",
        help="Record timings in memory but don't write a trace file.",
    )
    parser.add_argument(
        "--otlp-endpoint",
        default=None,
        dest="otlp_endpoint",
        help=(
            "Send the run's spans to an OpenTelemetry collector at this URL (e.g. "
            "http://localhost:4318/v1/traces). Needs the tracing extra: uv sync "
            "--all-packages --extra tracing. The JSON trace file is written too; "
            "omit this and the standard OTEL_EXPORTER_OTLP_ENDPOINT env var to skip "
            "the collector entirely."
        ),
    )
    parser.add_argument(
        "--sandbox",
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Where shell commands run. auto (the default) runs them in a container "
            "when Docker is available and on your machine when it isn't; on refuses "
            "to start without a container; off always runs them on your machine. "
            "Either way you're still asked before a command runs."
        ),
    )
    parser.add_argument(
        "--sandbox-image",
        default=DEFAULT_IMAGE,
        dest="sandbox_image",
        help="Container image for the sandbox (default: the shared project image).",
    )
    parser.add_argument(
        "--sandbox-network",
        action="store_true",
        dest="sandbox_network",
        help="Give the sandbox container network access. Off by default, on purpose.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0.0 is deterministic (the default)."
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Approve risky tools (write, delete, shell) without prompting. Use with care.",
    )
    parser.add_argument(
        "--ask",
        action="store_true",
        help="Prompt on the terminal for risky tools. This is the default; kept for compatibility.",
    )
    args = parser.parse_args()

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()

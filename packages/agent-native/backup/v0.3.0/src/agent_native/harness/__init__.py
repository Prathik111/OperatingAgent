"""Tool-calling reliability harness (decision #8).

Runs the agent's real tool schemas against a target LLM provider and measures
how often the model returns *parseable* tool calls with *schema-valid*
arguments and the *correct* tool for a given scenario.

    uv run --package agent-native agent-native-harness \
        --provider groq --model llama-3.3-70b-versatile --iterations 30

Provider keys: GROQ_API_KEY (env/.env) or a running Ollama instance.

The bundled scenario/schema fixture mirrors the file-server tool set
(packages/mcp-servers/file-server) so the harness measures the real contract
the executor depends on, not an idealized one.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from ..llm import LLMClient, build_llm
from ..types import ToolInfo, ToolSchema

_FIXTURE = "tool_schemas.json"


def _load_fixture() -> tuple[list[ToolInfo], list[dict[str, Any]]]:
    """Load bundled tool schemas + scenarios (kept in the package data dir)."""
    here = Path(__file__).resolve().parent
    data = json.loads((here / _FIXTURE).read_text(encoding="utf-8"))
    tools = [
        ToolInfo(
            name=t["name"],
            description=t["description"],
            schema=ToolSchema(input_schema=t.get("inputSchema", {}), output_schema={}),
            risk_level=t.get("riskLevel", "safe"),
        )
        for t in data["tools"]
    ]
    return tools, data["scenarios"]


def validate_args(arguments: dict, schema: dict) -> list[str]:
    """Minimal but real schema validation: required keys + primitive types."""
    errors: list[str] = []
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument {key!r}")
    for key, value in arguments.items():
        prop = props.get(key)
        if not prop:
            continue
        expected = prop.get("type")
        if expected == "string" and not isinstance(value, str):
            errors.append(f"{key!r} should be string, got {type(value).__name__}")
        elif expected == "integer" and not isinstance(value, int):
            errors.append(f"{key!r} should be integer, got {type(value).__name__}")
        elif expected == "boolean" and not isinstance(value, bool):
            errors.append(f"{key!r} should be boolean, got {type(value).__name__}")
        elif expected == "array" and not isinstance(value, list):
            errors.append(f"{key!r} should be array, got {type(value).__name__}")
        enum = prop.get("enum")
        if enum is not None and value not in enum:
            errors.append(f"{key!r} not in enum {enum}")
    return errors


async def run_trials(
    llm: LLMClient,
    tools: list[ToolInfo],
    scenarios: list[dict[str, Any]],
    iterations: int,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    stats = {
        "iterations": iterations,
        "parse_success": 0,
        "valid_arguments": 0,
        "correct_tool": 0,
        "tool_calls_emitted": 0,
        "per_tool": {},
        "failures": [],
    }
    for i in range(iterations):
        scenario = rng.choice(scenarios)
        expected_tool = scenario["tool"]
        response = await llm.complete(
            [{"role": "user", "content": scenario["prompt"]}],
            tools=tools,
        )
        calls = response.tool_calls
        if not calls:
            stats["failures"].append({"iteration": i, "kind": "no_tool_call", "scenario": scenario["tool"]})
            continue
        stats["tool_calls_emitted"] += 1
        call = calls[0]
        if call.parse_error or not call.arguments:
            stats["failures"].append({"iteration": i, "kind": "unparseable", "scenario": scenario["tool"]})
            continue
        stats["parse_success"] += 1
        tool = next((t for t in tools if t.name == call.name), None)
        if tool is None:
            stats["failures"].append({"iteration": i, "kind": "unknown_tool", "tool": call.name})
            continue
        if call.name == expected_tool:
            stats["correct_tool"] += 1
        errors = validate_args(call.arguments, tool.schema.input_schema)
        if errors:
            stats["failures"].append({"iteration": i, "kind": "invalid_args", "errors": errors,
                                      "tool": call.name, "scenario": scenario["tool"]})
            continue
        stats["valid_arguments"] += 1
        bucket = stats["per_tool"].setdefault(call.name, {"ok": 0, "total": 0})
        bucket["total"] += 1
        bucket["ok"] += 1
    return stats


def _report(stats: dict[str, Any], model: str) -> str:
    total = stats["iterations"]
    def pct(n: int) -> str:
        return f"{100.0 * n / total:.1f}%"
    lines = [
        f"harness report: model={model} iterations={total}",
        f"  parse success    : {stats['parse_success']}/{total} ({pct(stats['parse_success'])})",
        f"  valid arguments  : {stats['valid_arguments']}/{total} ({pct(stats['valid_arguments'])})",
        f"  correct tool     : {stats['correct_tool']}/{total} ({pct(stats['correct_tool'])})",
        f"  tool calls emitted: {stats['tool_calls_emitted']}/{total}",
    ]
    if stats["per_tool"]:
        lines.append("  per-tool parse:")
        for name, bucket in sorted(stats["per_tool"].items()):
            lines.append(f"    {name}: {bucket['ok']}/{bucket['total']}")
    for f in stats["failures"][:10]:
        lines.append(f"  fail: {f}")
    if stats["failures"]:
        lines.append(f"  total failures: {len(stats['failures'])} (shown up to 10)")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    tools, scenarios = _load_fixture()
    settings = _settings_for(args)
    try:
        llm = build_llm(settings)
    except ValueError as e:
        print(f"harness error: {e}", file=sys.stderr)
        return 1
    print(f"running {args.iterations} trials against {args.provider}:{args.model} "
          f"with {len(tools)} tools", file=sys.stderr)
    stats = await run_trials(llm, tools, scenarios, args.iterations, seed=args.seed)
    print(_report(stats, args.model))
    valid = stats["parse_success"] / max(1, stats["iterations"])
    return 0 if valid >= args.threshold else 1


def _settings_for(args: argparse.Namespace):
    from ..config import Settings

    settings = Settings()
    settings.llm_provider = args.provider
    if args.provider == "groq":
        settings.groq_model = args.model
    else:
        settings.ollama_model = args.model
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-native-harness", description=__doc__)
    parser.add_argument("--provider", choices=["groq", "ollama"], default="groq")
    parser.add_argument("--model", default=None, help="defaults per provider")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="minimum parse rate for exit code 0 (0..1)")
    args = parser.parse_args()
    if args.model is None:
        args.model = ("llama-3.3-70b-versatile" if args.provider == "groq" else "llama3.1:8b")
    sys.exit(asyncio_run(main_async(args)))


def asyncio_run(awaitable) -> int:
    import asyncio

    return asyncio.run(awaitable)


if __name__ == "__main__":
    main()

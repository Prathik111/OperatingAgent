"""LLM layer and harness core tests (provider clients stay SDK-mocked)."""

from __future__ import annotations

import json

import pytest

from agent_native.harness import _load_fixture, run_trials, validate_args
from agent_native.llm import LLMResponse, ToolCall, Usage, _parse_arguments
from conftest import FakeLLM  # noqa: F401


def test_parse_arguments_handles_garbage():
    assert _parse_arguments(None) == {}
    assert _parse_arguments("not json") == {}
    assert _parse_arguments("[1,2]") == {}
    assert _parse_arguments('{"a": 1}') == {"a": 1}


def test_harness_fixture_loads():
    tools, scenarios = _load_fixture()
    assert len(tools) >= 6
    names = {t.name for t in tools}
    assert {"read_file", "write_file", "list_directory"} <= names
    assert scenarios and all(s.get("tool") and s.get("prompt") for s in scenarios)


class ScenarioAwareLLM(FakeLLM):
    """Responds with the correct tool+args for whatever scenario prompt it
    receives (order-independent, unlike a fixed script)."""

    KEYWORDS = {
        "Read the contents": "read_file",
        "Create a new file": "write_file",
        "List what files": "list_directory",
        "Check whether": "exists",
        "Find every .md": "search_files",
        "size and last-modified": "metadata",
        "Rename": "move_file",
        "Delete the leftover": "delete_file",
    }

    async def complete(self, messages, tools=None, *, temperature=0.0):
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        prompt = messages[0]["content"] if messages else ""
        for keyword, tool in self.KEYWORDS.items():
            if keyword.lower() in prompt.lower():
                return LLMResponse(
                    tool_calls=[ToolCall(id="x", name=tool, arguments=valid_args_for(tool))],
                    usage=Usage(1, 1),
                )
        return LLMResponse(text="i refuse", usage=Usage(1, 1))


@pytest.mark.asyncio
async def test_run_trials_counts_perfect_model():
    tools, scenarios = _load_fixture()
    llm = ScenarioAwareLLM([])
    stats = await run_trials(llm, tools, scenarios[:2], iterations=2, seed=0)
    assert stats["parse_success"] == 2
    assert stats["valid_arguments"] == 2
    assert stats["correct_tool"] == 2


def valid_args_for(tool: str) -> dict:
    return {
        "read_file": {"path": "/work/a.txt"},
        "write_file": {"path": "/work/a.txt", "content": "hi"},
        "list_directory": {"path": "/work"},
        "exists": {"path": "/work/a.txt"},
        "search_files": {"path": "/work", "pattern": "*.md"},
        "metadata": {"path": "/work/a.txt"},
        "move_file": {"source": "/work/a.txt", "destination": "/work/b.txt"},
        "delete_file": {"path": "/work/a.txt"},
    }[tool]


@pytest.mark.asyncio
async def test_run_trials_counts_unparseable_and_wrong_tool():
    tools, scenarios = _load_fixture()

    class FlakyLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__([])
            self.n = 0

        async def complete(self, messages, tools=None, *, temperature=0.0):
            self.n += 1
            prompt = messages[0]["content"] if messages else ""
            flaky = self.n == 3
            if flaky:
                return LLMResponse(text="i refuse", usage=Usage(1, 1))  # no tool call
            for keyword, tool in ScenarioAwareLLM.KEYWORDS.items():
                if keyword.lower() in prompt.lower():
                    if self.n == 2:
                        args = {"path": 42}  # wrong type -> schema-invalid but parseable
                    else:
                        args = valid_args_for(tool)
                    return LLMResponse(tool_calls=[ToolCall(id="x", name=tool, arguments=args)],
                                       usage=Usage(1, 1))
            return LLMResponse(text="i refuse", usage=Usage(1, 1))

    llm = FlakyLLM()
    stats = await run_trials(llm, tools, scenarios[:3], iterations=3, seed=1)
    assert stats["parse_success"] == 2
    assert stats["valid_arguments"] == 1  # 2nd call: missing required argument
    assert stats["correct_tool"] == 2


def test_validate_args_required_and_types():
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "count": {"type": "integer"},
            "mode": {"enum": ["r", "w"]},
        },
        "required": ["path", "mode"],
    }
    assert validate_args({"path": "p", "mode": "r"}, schema) == []
    errors = validate_args({"path": 3, "mode": "x"}, schema)
    assert any("path" in e for e in errors)
    assert any("enum" in e for e in errors)
    missing = validate_args({"path": "p"}, schema)
    assert any("mode" in e for e in missing)


def test_fixture_json_well_formed():
    tools, scenarios = _load_fixture()
    for tool in tools:
        assert isinstance(tool.schema.input_schema, dict)
        assert isinstance(tool.schema.input_schema.get("required"), list)
        assert json.dumps(tool.schema.input_schema)  # serializable
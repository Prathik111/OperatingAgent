"""Unit tests for ``PromptManager`` (``agent_langgraph.runtime.prompt_manager``).

Loads each node's system prompt from a directory of ``*.txt`` files. The missing
file case matters: a bad override must get a clear ``FileNotFoundError`` naming
the file rather than an empty prompt silently reaching the model.
"""

from __future__ import annotations

import pytest
from agent_langgraph.runtime.prompt_manager import DEFAULT_PROMPT_DIR, PromptManager
from common.config import PromptConfig


@pytest.fixture
def prompt_dir(tmp_path):
    (tmp_path / "planner.txt").write_text("PLAN PROMPT", encoding="utf-8")
    (tmp_path / "verifier.txt").write_text("VERIFY PROMPT", encoding="utf-8")
    (tmp_path / "responder.txt").write_text("RESPOND PROMPT", encoding="utf-8")
    return tmp_path


def test_prompt_manager_loads_each_prompt(prompt_dir) -> None:
    manager = PromptManager(prompt_dir)
    assert manager.planner() == "PLAN PROMPT"
    assert manager.verifier() == "VERIFY PROMPT"
    assert manager.responder() == "RESPOND PROMPT"


def test_prompt_manager_accepts_str_path(prompt_dir) -> None:
    manager = PromptManager(str(prompt_dir))
    assert manager.planner() == "PLAN PROMPT"


def test_packaged_default_prompts_are_available() -> None:
    manager = PromptManager(DEFAULT_PROMPT_DIR)
    assert manager.planner().strip()
    assert manager.verifier().strip()
    assert manager.responder().strip()


def test_prompt_manager_honors_each_exact_configured_path(tmp_path) -> None:
    planner = tmp_path / "planner-custom.md"
    verifier_dir = tmp_path / "verify"
    responder_dir = tmp_path / "respond"
    verifier_dir.mkdir()
    responder_dir.mkdir()
    verifier = verifier_dir / "system.txt"
    responder = responder_dir / "answer.txt"
    planner.write_text("P", encoding="utf-8")
    verifier.write_text("V", encoding="utf-8")
    responder.write_text("R", encoding="utf-8")

    manager = PromptManager(PromptConfig(planner, verifier, responder))

    assert manager.planner() == "P"
    assert manager.verifier() == "V"
    assert manager.responder() == "R"


@pytest.mark.regression
def test_prompt_manager_missing_file_raises(tmp_path) -> None:
    """Never degrade to an empty prompt — name the file that is missing."""
    manager = PromptManager(tmp_path)  # empty dir
    with pytest.raises(FileNotFoundError, match="planner.txt"):
        manager.planner()

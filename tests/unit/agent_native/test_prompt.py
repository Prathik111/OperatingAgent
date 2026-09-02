"""What the agent is told before it does anything: the first prompt, and the
part of every request that must not move.

Two separate concerns that both live at the front of a request, which is why
they're tested together:

  * the system prompt should say where the agent is, so the model doesn't spend
    its first turn finding out (config.py);
  * whatever it says, it has to say the *same* thing every turn or the provider's
    prompt cache can never hit (models/base.py).

No network and no key: everything here is a pure function of its inputs.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from agent_native.config import (
    AgentConfig,
    PromptBuilder,
    read_branch,
    read_folder_listing,
)
from agent_native.models.base import mark_cacheable_prefix, stable_prefix_fingerprint


def _workdir(*, files=(), folders=(), head: str = "") -> str:
    workdir = tempfile.mkdtemp()
    for folder in folders:
        os.makedirs(os.path.join(workdir, folder), exist_ok=True)
    for name in files:
        open(os.path.join(workdir, name), "w").close()
    if head:
        os.makedirs(os.path.join(workdir, ".git"), exist_ok=True)
        with open(os.path.join(workdir, ".git", "HEAD"), "w") as fh:
            fh.write(head)
    return workdir


# -- what the folder listing shows, and what it doesn't ----------------------
async def test_listing_puts_folders_first_and_skips_noise():
    """Folders first, then files, each in the order a person would read them -
    case-insensitively, so `main.py` sorts before `README.md`. `__pycache__` is
    not information about the project."""
    workdir = _workdir(files=("README.md", "main.py"), folders=("src", "__pycache__"))
    assert read_folder_listing(workdir) == "src/, main.py, README.md"


async def test_listing_leaves_out_hidden_files():
    """A `.env` holds the API key. It is not a name to volunteer to a model."""
    workdir = _workdir(files=("README.md", ".env"))
    listing = read_folder_listing(workdir)
    assert ".env" not in listing and listing == "README.md"


async def test_listing_is_capped_and_says_so():
    workdir = _workdir(files=[f"f{i:02d}.txt" for i in range(40)])
    listing = read_folder_listing(workdir)
    assert listing.count(",") == 29        # 30 names shown
    assert "(+10 more)" in listing         # and it admits to the rest


async def test_a_missing_folder_is_silence_not_an_exception():
    assert read_folder_listing(os.path.join(tempfile.mkdtemp(), "nope")) == ""


# -- the branch, read straight from .git, with no subprocess -----------------
async def test_branch_is_read_from_the_head_file():
    workdir = _workdir(head="ref: refs/heads/feature/scoped-grants\n")
    assert read_branch(workdir) == "feature/scoped-grants"


async def test_a_detached_head_says_so():
    workdir = _workdir(head="a" * 40 + "\n")
    assert read_branch(workdir) == "detached at aaaaaaaa"


async def test_no_repo_is_silence_not_an_exception():
    assert read_branch(tempfile.mkdtemp()) == ""


async def test_both_reach_the_system_prompt():
    workdir = _workdir(files=("README.md",), folders=("src",), head="ref: refs/heads/main\n")
    prompt = PromptBuilder().build(
        AgentConfig(),
        SimpleNamespace(working_directory=workdir),
        ["filesystem_read_file"],
    )
    assert "Git branch: main" in prompt
    assert "At the top level of it: src/, README.md" in prompt


# -- the part of the request that must not move ------------------------------
_TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {}}}]


async def test_fingerprint_ignores_everything_after_the_system_messages():
    """Two turns of one run differ in their tail. That must not change the front."""
    turn_one = [{"role": "system", "content": "you are helpful"}, {"role": "user", "content": "hi"}]
    turn_two = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "something else entirely"},
        {"role": "assistant", "content": "and a reply"},
    ]
    assert stable_prefix_fingerprint(turn_one, _TOOLS) == stable_prefix_fingerprint(turn_two, _TOOLS)


async def test_a_date_in_the_system_prompt_changes_the_fingerprint():
    """The failure this exists to catch: a prefix that quietly moves every turn."""
    steady = [{"role": "system", "content": "you are helpful"}]
    moved = [{"role": "system", "content": "you are helpful. Today is Tuesday"}]
    assert stable_prefix_fingerprint(steady, _TOOLS) != stable_prefix_fingerprint(moved, _TOOLS)


async def test_changing_the_tools_changes_the_fingerprint():
    steady = [{"role": "system", "content": "you are helpful"}]
    assert stable_prefix_fingerprint(steady, _TOOLS) != stable_prefix_fingerprint(
        steady, _TOOLS + _TOOLS
    )


async def test_no_marker_means_the_messages_are_untouched():
    """Every model in this repo. An empty marker must not rewrite the request."""
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
    assert mark_cacheable_prefix(messages, "") is messages


async def test_a_marker_lands_on_the_last_leading_system_message_only():
    marked = mark_cacheable_prefix(
        [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "c"},
        ],
        "cache_control",
    )
    assert marked[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in marked[0]      # one marker is enough
    assert marked[2] == {"role": "user", "content": "c"}

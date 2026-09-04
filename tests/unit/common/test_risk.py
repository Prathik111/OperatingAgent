"""Tests for the deterministic tool risk policy (``common.risk``).

The classifier is a security control — the executor consults it to decide
whether a human gate is required — so these tests pin both the verdicts and the
ordering guarantee that makes the policy fail safe.
"""

from __future__ import annotations

import pytest
from common.enums import RiskLevel
from common.risk import DEFAULT_RULES, RiskClassifier, RiskRule
from common.tools import ToolCallRequest


@pytest.fixture
def classifier() -> RiskClassifier:
    return RiskClassifier()


def call(tool_name: str, **arguments: object) -> ToolCallRequest:
    return ToolCallRequest(tool_name=tool_name, arguments=dict(arguments))


# ---------------------------------------------------------------------------
# BLOCKED — destructive / unrecoverable
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("terminal_run", {"command": "rm -rf /"}),
        ("terminal_run", {"command": "rm  -rf ./build"}),
        ("terminal_run", {"command": "mkfs.ext4 /dev/sda"}),
        ("disk_format", {"target": "C:"}),
        ("db_query", {"sql": "DROP TABLE users"}),
        ("db_query", {"sql": "drop database prod"}),
        ("git_push", {"args": "push --force origin main"}),
        ("terminal_run", {"command": "sudo apt install evil"}),
        ("terminal_run", {"command": ":(){ :|:& };:"}),
    ],
)
def test_blocked_calls(classifier: RiskClassifier, tool_name: str, arguments: dict) -> None:
    assert classifier.classify(call(tool_name, **arguments)) is RiskLevel.BLOCKED


@pytest.mark.regression
@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /tmp/x",                    # canonical
        "rm -fr /tmp/x",                    # flags reversed in one cluster
        "rm -r -f /tmp/x",                  # split short flags
        "rm -f -r /tmp/x",                  # split, reversed order
        "rm --recursive --force /tmp/x",    # long flags
        "rm --force --recursive /tmp/x",    # long flags, reversed
        "rm -r --force /tmp/x",             # mixed short + long
        "rm --recursive -f /tmp/x",         # mixed long + short
        "rm -rfv /tmp/x",                   # cluster with a trailing verbose flag
        "rm -frv /tmp/x",                   # cluster, r/f reordered, extra letter
        "rm -vrf /tmp/x",                   # cluster, r/f not adjacent
    ],
)
def test_recursive_force_rm_variants_are_blocked(
    classifier: RiskClassifier, command: str
) -> None:
    """Every spelling of a recursive+force delete must be BLOCKED, not just the
    single ``rm -rf`` form."""
    assert classifier.classify(call("terminal_run", command=command)) is RiskLevel.BLOCKED


@pytest.mark.regression
def test_recursive_rm_without_force_is_not_blocked(classifier: RiskClassifier) -> None:
    """Recursion alone is not the catastrophic case the BLOCKED rule targets; it
    still trips REVIEW via the ``terminal`` tool name, but must not be BLOCKED."""
    assert classifier.classify(call("terminal_run", command="rm -r /tmp/x")) is RiskLevel.REVIEW


@pytest.mark.regression
@pytest.mark.parametrize(
    "command",
    [
        "rm -f /tmp/x",                       # force alone
        "rm --interactive --force /tmp/x",    # 'interactive' contains 'r' but isn't recursive
        "rm --verbose --force /tmp/x",        # 'verbose' contains 'r' but isn't recursive
    ],
)
def test_force_rm_without_recursive_is_not_blocked(
    classifier: RiskClassifier, command: str
) -> None:
    """Force without recursion is not the catastrophic case. The broadened
    short-cluster match must not read the incidental ``r`` in a long flag such
    as ``--interactive``/``--verbose`` as a recursive flag; these stay REVIEW
    (via the ``terminal`` tool name) but must not be BLOCKED."""
    assert classifier.classify(call("terminal_run", command=command)) is RiskLevel.REVIEW


# ---------------------------------------------------------------------------
# REVIEW — mutating but recoverable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("filesystem_delete_file", {"path": "notes.txt"}),
        ("filesystem_write_file", {"path": "a.txt", "content": "x"}),
        ("filesystem_move_file", {"src": "a", "dst": "b"}),
        ("terminal_run", {"command": "echo hi"}),  # matches 'command'/'terminal'
        ("git_commit", {"message": "wip"}),
        ("http_fetch", {"url": "https://example.com"}),
        ("deps_install", {"package": "requests"}),
    ],
)
def test_review_calls(classifier: RiskClassifier, tool_name: str, arguments: dict) -> None:
    assert classifier.classify(call(tool_name, **arguments)) is RiskLevel.REVIEW


# ---------------------------------------------------------------------------
# SAFE — nothing matched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("filesystem_read_file", {"path": "notes.txt"}),
        ("filesystem_list_directory", {"path": "."}),
        ("git_status", {}),
        ("search_documents", {"query": "hello"}),
        ("filesystem_exists", {"path": "a.txt"}),
    ],
)
def test_safe_calls(classifier: RiskClassifier, tool_name: str, arguments: dict) -> None:
    assert classifier.classify(call(tool_name, **arguments)) is RiskLevel.SAFE


@pytest.mark.regression
@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        # 'input'/'output' as parameter *names* must not read as a network 'put'
        # now that keys are excluded from the haystack.
        ("data_transform", {"input": "a.csv", "output": "b.csv"}),
        # ...and 'put' inside the values 'input.csv'/'output.csv' must not match
        # now that the network tokens are word-bounded.
        ("data_transform", {"src": "input.csv", "dst": "output.csv"}),
        # Benign free text that merely mentions concepts (no risky standalone
        # token: 'formatting' is not 'format', 'output' is not 'put').
        ("search_documents", {"query": "quarterly revenue summary"}),
        ("search_documents", {"query": "output formatting tips"}),
    ],
)
def test_benign_payloads_are_safe(
    classifier: RiskClassifier, tool_name: str, arguments: dict
) -> None:
    assert classifier.classify(call(tool_name, **arguments)) is RiskLevel.SAFE


# ---------------------------------------------------------------------------
# Ordering / matching behaviour
# ---------------------------------------------------------------------------


def test_separator_normalised_names_match_word_boundaries(classifier: RiskClassifier) -> None:
    """``git_push`` / ``fs.delete`` must match ``\\bpush\\b`` / ``delete`` even
    though the separators would otherwise defeat the word boundary."""
    assert classifier.classify(call("fs.delete", path="x")) is RiskLevel.REVIEW
    # push+force spans the normalised form; severity ordering must still pick BLOCKED.
    assert classifier.classify(call("git_push", args="force")) is RiskLevel.BLOCKED


@pytest.mark.regression
def test_severity_ordering_blocked_beats_review(classifier: RiskClassifier) -> None:
    """A call matching both a BLOCKED and a REVIEW rule resolves to BLOCKED,
    because rules are scanned most-dangerous-first."""
    request = call("terminal_run", command="sudo rm file")  # 'sudo' BLOCKED, 'rm'/'remove' REVIEW-ish
    assert classifier.classify(request) is RiskLevel.BLOCKED


def test_matching_is_case_insensitive(classifier: RiskClassifier) -> None:
    assert classifier.classify(call("DB", sql="DROP TABLE X")) is RiskLevel.BLOCKED
    assert classifier.classify(call("FS", cmd="DELETE")) is RiskLevel.REVIEW


@pytest.mark.regression
def test_arguments_are_inspected_not_just_the_name(classifier: RiskClassifier) -> None:
    """A benign tool name with a dangerous argument is still classified on the
    argument — the haystack includes the stringified arguments."""
    assert classifier.classify(call("run", command="rm -rf /")) is RiskLevel.BLOCKED


@pytest.mark.regression
def test_set_arguments_are_rejected_for_determinism(classifier: RiskClassifier) -> None:
    """A set has no defined iteration order, so it could reorder the haystack and
    flip a multi-token verdict between runs. The classifier is a security control
    and must be deterministic, so it refuses a set rather than classify an
    arbitrary ordering — the caller must pass an ordered sequence."""
    with pytest.raises(TypeError):
        classifier.classify(call("terminal_run", command={"rm", "-rf", "/"}))
    with pytest.raises(TypeError):
        classifier.classify(call("terminal_run", command=frozenset({"rm", "-rf", "/"})))


def test_ordered_sequence_arguments_are_inspected(classifier: RiskClassifier) -> None:
    """List and tuple arguments remain supported and contribute their elements to
    the haystack, so a dangerous command split across a sequence is still seen."""
    assert classifier.classify(call("run", parts=["rm", "-rf", "/"])) is RiskLevel.BLOCKED
    assert classifier.classify(call("run", parts=("rm", "-rf", "/"))) is RiskLevel.BLOCKED


def test_explain_returns_level_and_reason(classifier: RiskClassifier) -> None:
    level, reason = classifier.explain(call("db", sql="DROP TABLE t"))
    assert level is RiskLevel.BLOCKED
    assert reason == "destructive SQL"


def test_explain_defaults_to_safe_with_reason(classifier: RiskClassifier) -> None:
    level, reason = classifier.explain(call("read_file", path="a.txt"))
    assert level is RiskLevel.SAFE
    assert "no risk pattern matched" in reason


def test_classify_delegates_to_explain(classifier: RiskClassifier) -> None:
    request = call("db", sql="DROP TABLE t")
    assert classifier.classify(request) is classifier.explain(request)[0]


# ---------------------------------------------------------------------------
# Custom rule sets
# ---------------------------------------------------------------------------


def test_custom_rules_replace_defaults() -> None:
    """A caller-supplied rule set is honoured, and unmatched calls fall to SAFE."""
    rules = (RiskRule(r"\blaunch\b", RiskLevel.BLOCKED, "missile launch"),)
    classifier = RiskClassifier(rules)
    assert classifier.classify(call("launch_now")) is RiskLevel.BLOCKED
    # A default-blocked pattern is no longer present in the custom set.
    assert classifier.classify(call("run", command="rm -rf /")) is RiskLevel.SAFE


@pytest.mark.regression
def test_default_rules_are_ordered_most_dangerous_first() -> None:
    """The first BLOCKED rule must precede the first REVIEW rule, since first
    match wins — this ordering is the invariant the policy relies on."""
    levels = [rule.level for rule in DEFAULT_RULES]
    first_review = levels.index(RiskLevel.REVIEW)
    assert all(level is RiskLevel.BLOCKED for level in levels[:first_review])


def test_empty_ruleset_classifies_everything_safe() -> None:
    classifier = RiskClassifier(rules=())
    assert classifier.classify(call("run", command="rm -rf /")) is RiskLevel.SAFE

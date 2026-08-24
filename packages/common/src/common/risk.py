"""Deterministic tool risk policy.

RiskClassifier maps a tool call to a RiskLevel using an ordered list of
pattern rules — no LLM, no side effects — so the same call always yields the
same verdict. The ExecutorNode / ApprovalGateway consult it before running a
tool to decide whether a human gate is required.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .enums import RiskLevel
from .tools import ToolCallRequest


@dataclass(slots=True, frozen=True)
class RiskRule:
    """One classification rule. The first matching rule (highest risk first)
    wins. ``pattern`` is matched case-insensitively against the tool name and
    the stringified argument *values* (dict keys are excluded — see
    ``_haystacks``)."""

    pattern: str
    level: RiskLevel
    reason: str


# Ordered most-dangerous first so the first match is the strongest verdict.
# Patterns are matched against both the raw text and a separator-normalised
# form (see _haystacks), so tool names like "git_push" match \bpush\b.
DEFAULT_RULES: tuple[RiskRule, ...] = (
    # --- BLOCKED: destructive or unrecoverable ---------------------------
    # Recursive-force delete in any flag spelling. After separator
    # normalisation every flag becomes a bare token (``-rf`` -> ``rf``,
    # ``--recursive`` -> ``recursive``), so the two lookaheads just require a
    # recursive token AND a force token somewhere after ``rm`` — covering
    # ``rm -rf``, ``rm -fr``, ``rm -r -f`` and ``rm --recursive --force``.
    RiskRule(
        r"\brm\b(?=.*\b(?:rf|fr|r|recursive)\b)(?=.*\b(?:rf|fr|f|force)\b)",
        RiskLevel.BLOCKED,
        "recursive force delete",
    ),
    RiskRule(r"\bmkfs\b|\bformat\b", RiskLevel.BLOCKED, "filesystem format"),
    RiskRule(r"\bdrop\s+(table|database)\b", RiskLevel.BLOCKED, "destructive SQL"),
    RiskRule(r"\bpush\b.*\bforce\b", RiskLevel.BLOCKED, "force push"),
    RiskRule(r"\bsudo\b", RiskLevel.BLOCKED, "privilege escalation"),
    RiskRule(r":\(\)\s*\{.*\};:", RiskLevel.BLOCKED, "fork bomb"),
    # --- REVIEW: mutating but recoverable --------------------------------
    RiskRule(r"delete|remove|unlink|truncate", RiskLevel.REVIEW, "deletion"),
    RiskRule(r"write|edit|create|update|move|rename|chmod|chown", RiskLevel.REVIEW, "filesystem mutation"),
    RiskRule(r"shell|terminal|exec|subprocess|command|bash|powershell", RiskLevel.REVIEW, "arbitrary execution"),
    RiskRule(r"\bgit\b.*(commit|push|reset|rebase|merge)", RiskLevel.REVIEW, "git state change"),
    # Network tokens are word-bounded so benign payloads don't trip them:
    # ``put`` must not match inside ``input``/``output``, nor ``post`` inside a
    # filename. ``https?`` keeps both http and https.
    RiskRule(
        r"\bhttps?\b|\brequest\b|\bfetch\b|\bcurl\b|\bwget\b|\bpost\b|\bput\b",
        RiskLevel.REVIEW,
        "network access",
    ),
    RiskRule(r"install|pip|npm|apt|uv\s+add", RiskLevel.REVIEW, "dependency change"),
)

# Characters that separate words in tool names but defeat \b boundaries.
_SEPARATORS = re.compile(r"[^0-9A-Za-z]+")


class RiskClassifier:
    """Deterministic pattern-based risk policy.

    Matches the documented contract: ``classify(call) -> RiskLevel``. Rules are
    compiled once at construction; classification is pure and allocation-light.
    """

    def __init__(self, rules: tuple[RiskRule, ...] = DEFAULT_RULES) -> None:
        self.patterns: list[tuple[re.Pattern[str], RiskLevel, str]] = [
            (re.compile(rule.pattern, re.IGNORECASE), rule.level, rule.reason)
            for rule in rules
        ]

    def classify(self, call: ToolCallRequest) -> RiskLevel:
        """Return the risk level for a tool call.

        Inspects the tool name and its argument *values* (not the dict keys —
        parameter names are structural, not payload); the first matching rule
        (rules are ordered most-dangerous first) determines the verdict.
        Defaults to SAFE when nothing matches.
        """
        return self.explain(call)[0]

    def explain(self, call: ToolCallRequest) -> tuple[RiskLevel, str]:
        """Like ``classify`` but also returns the reason, for audit/logging."""
        texts = _haystacks(call)
        # Patterns outermost so severity order wins: a BLOCKED rule matching the
        # normalised form must beat a REVIEW rule matching the raw form.
        for pattern, level, reason in self.patterns:
            if any(pattern.search(text) for text in texts):
                return level, reason
        return RiskLevel.SAFE, "no risk pattern matched"


def _iter_arg_values(value: Any) -> Iterator[str]:
    """Yield the stringified *values* of an argument tree, ignoring dict keys.

    Parameter names (dict keys) are structural — they come from a tool's
    schema, not from the payload — so matching against them produces false
    positives: a benign ``{"output": ...}`` key would read as a network ``put``
    and ``{"format": ...}`` as a disk format. Only the values carry the command
    text worth classifying, so only they go into the haystack.
    """
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_arg_values(nested)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_arg_values(item)
    else:
        yield str(value)


def _haystacks(call: ToolCallRequest) -> tuple[str, str]:
    """The raw text and a separator-normalised copy.

    The text is the tool name plus the flattened argument *values* (never the
    keys). Matching both forms means punctuation-sensitive patterns (``rm -rf``,
    fork bombs) still work on the raw form, while word-boundary patterns also
    catch underscore/dot-separated tool names such as ``git_push`` or
    ``fs.delete``.
    """
    raw = " ".join([call.tool_name, *_iter_arg_values(call.arguments)])
    return raw, _SEPARATORS.sub(" ", raw)

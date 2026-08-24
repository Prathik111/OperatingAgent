"""Deterministic tool risk policy.

RiskClassifier maps a tool call to a RiskLevel using an ordered list of
pattern rules — no LLM, no side effects — so the same call always yields the
same verdict. The ExecutorNode / ApprovalGateway consult it before running a
tool to decide whether a human gate is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .enums import RiskLevel
from .tools import ToolCallRequest


@dataclass(slots=True, frozen=True)
class RiskRule:
    """One classification rule. The first matching rule (highest risk first)
    wins. ``pattern`` is matched case-insensitively against the tool name and
    the stringified arguments."""

    pattern: str
    level: RiskLevel
    reason: str


# Ordered most-dangerous first so the first match is the strongest verdict.
# Patterns are matched against both the raw text and a separator-normalised
# form (see _haystacks), so tool names like "git_push" match \bpush\b.
DEFAULT_RULES: tuple[RiskRule, ...] = (
    # --- BLOCKED: destructive or unrecoverable ---------------------------
    RiskRule(r"\brm\s+-?rf\b", RiskLevel.BLOCKED, "recursive force delete"),
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
    RiskRule(r"http|request|fetch|curl|wget|post|put", RiskLevel.REVIEW, "network access"),
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

        Inspects both the tool name and its arguments; the first matching rule
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


def _haystacks(call: ToolCallRequest) -> tuple[str, str]:
    """The raw text and a separator-normalised copy.

    Matching both means punctuation-sensitive patterns (``rm -rf``, fork bombs)
    still work on the raw form, while word-boundary patterns also catch
    underscore/dot-separated tool names such as ``git_push`` or ``fs.delete``.
    """
    raw = f"{call.tool_name} {call.arguments}"
    return raw, _SEPARATORS.sub(" ", raw)

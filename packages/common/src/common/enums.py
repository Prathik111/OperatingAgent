from enum import Enum


class AgentTrack(str, Enum):
    NATIVE = "native"
    LANGGRAPH = "langgraph"


class RunStatus(str, Enum):
    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RiskLevel(str, Enum):
    SAFE = "safe"
    REVIEW = "review"
    BLOCKED = "blocked"

class VerificationResult(str, Enum):
    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"
    SKIPPED = "skipped"

class TaskStatus(str, Enum):
        PLANNING = "planning"
        EXECUTING = "executing"
        VERIFYING = "verifying"
        RESPONDING = "responding"
        COMPLETED = "completed"
        FAILED = "failed"
        SKIPPED = "skipped"
        INTERRUPTED = "interrupted"


class WorkflowPhase(str, Enum):
    """Coarse stage of a multi-phase task.

    A task like "check the workspace for bugs" needs two distinct plans: first a
    read-only investigation, then a remediation built from what it found. Phases
    advance monotonically (investigate -> remediate -> complete), which is what
    bounds the number of replans and guarantees the graph terminates.
    """

    INVESTIGATE = "investigate"
    REMEDIATE = "remediate"
    COMPLETE = "complete"
        
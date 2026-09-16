from typing import Literal

# 1. Define the actual Node/Route types using Literal
# This tells Pylance exactly what string values are allowed.
NodeType = Literal[
    "planner", "executor", "verifier", "responder", "error_handler",
    "phase_transition",
]

# 2. Assign your string constants (can be used as values at runtime)
PLANNER: NodeType = "planner"
EXECUTOR: NodeType = "executor"
VERIFIER: NodeType = "verifier"
RESPONDER: NodeType = "responder"
ERROR_HANDLER: NodeType = "error_handler"
PHASE_TRANSITION: NodeType = "phase_transition"

ROUTE_EXECUTE: NodeType = EXECUTOR
ROUTE_VERIFY: NodeType = VERIFIER
ROUTE_RESPOND: NodeType = RESPONDER
ROUTE_ERROR: NodeType = ERROR_HANDLER
ROUTE_RETRY: NodeType = PLANNER
ROUTE_PHASE: NodeType = PHASE_TRANSITION

#: Fallback replan budget for states that predate the configurable
#: ``execution.max_replans`` (checkpoints, bare test states). Live runs carry
#: their budget on the state instead; the orchestrator seeds it.
MAX_RETRIES: int = 3
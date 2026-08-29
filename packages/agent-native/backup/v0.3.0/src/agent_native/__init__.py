"""OperatingAgent native agent (Plan-and-Execute + ReAct, hand-written).

Fully self-contained: no imports from packages/common. Own types, own
protocols, own repository.
"""

__version__ = "0.3.0"

from .types import (
    AgentRunResult,
    AgentTask,
    ApprovalDecision,
    Plan,
    PlanStep,
    RiskLevel,
    RunStatus,
    StepKind,
    StepOutcomeStatus,
    StepStatus,
    ToolCallRequest,
    ToolCallResult,
    ToolInfo,
    ToolSchema,
)
from .events import AgentEvent
from .agent import NativeAgent, build_agent, new_task
from .config import Settings, load_settings
from .planner import Planner, PlanningError
from .executor import ReactExecutor
from .reflector import Reflector, ReplanBudgetExhausted
from .verifier import Verifier, VerificationOutcome
from .risk import RiskClassifier
from .approval import ApprovalGateway
from .sandbox import SandboxManager
from .mcp import MCPClient, StdioMCPClient
from .repository import InMemoryTaskRepository, PostgresTaskRepository, TaskRepository
from .compactor import ContextCompactor
from .llm import GroqLLMClient, OllamaLLMClient, LLMClient

__all__ = [
    "AgentRunResult", "AgentTask", "ApprovalDecision", "Plan", "PlanStep",
    "RiskLevel", "RunStatus", "StepKind", "StepOutcomeStatus", "StepStatus",
    "ToolCallRequest", "ToolCallResult", "ToolInfo", "ToolSchema",
    "AgentEvent", "NativeAgent", "build_agent", "new_task",
    "Settings", "load_settings", "Planner", "PlanningError", "ReactExecutor",
    "Reflector", "ReplanBudgetExhausted", "Verifier", "VerificationOutcome",
    "RiskClassifier", "ApprovalGateway", "SandboxManager", "MCPClient",
    "StdioMCPClient", "InMemoryTaskRepository", "PostgresTaskRepository",
    "TaskRepository", "ContextCompactor", "GroqLLMClient", "OllamaLLMClient",
    "LLMClient", "__version__",
]
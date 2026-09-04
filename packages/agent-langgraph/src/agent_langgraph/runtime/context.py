from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from common.approvals import ApprovalHandler
from common.config import AgentConfig
from common.events import AgentEvent
from common.risk import RiskClassifier
from common.tools import ToolCallResult, ToolInfo

from ..tracing.tracer import Tracer


class ModelProviderLike(Protocol):
    def get_model(self) -> Any: ...


class ToolRegistryLike(Protocol):
    async def list_tools(self) -> list[ToolInfo]: ...

    async def call_by_name(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        workspace: str | None = None,
    ) -> ToolCallResult: ...


class PromptManagerLike(Protocol):
    def planner(self) -> str: ...

    def verifier(self) -> str: ...

    def responder(self) -> str: ...


@dataclass(frozen=True, slots=True)
class AgentContext:
    """
    Application dependencies available to LangGraph nodes.

    This object is NOT graph state and is NOT checkpointed.
    """

    model_provider: ModelProviderLike
    tool_registry: ToolRegistryLike
    risk_classifier: RiskClassifier
    prompt_manager: PromptManagerLike
    tracer: Tracer
    config: AgentConfig
    approval_handler: ApprovalHandler | None = None
    task_id: str = "standalone"
    event_sink: Callable[[AgentEvent], Awaitable[None] | None] | None = None
    completed_tool_calls: dict[str, str] | None = None
    workspace: str | None = None

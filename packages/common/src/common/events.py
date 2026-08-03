from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentEvent:

    type: str

    payload: dict[str, Any]


@dataclass(slots=True)
class PlanningStarted(AgentEvent):

    pass


@dataclass(slots=True)
class ToolStarted(AgentEvent):

    pass


@dataclass(slots=True)
class ToolFinished(AgentEvent):

    pass


@dataclass(slots=True)
class AgentFinished(AgentEvent):

    pass
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class AgentEvent:

    type: str

    payload: dict[str, Any]


def _payload(data: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, datetime): return value.isoformat()
        if isinstance(value, dict): return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list): return [convert(v) for v in value]
        return value
    return {key: convert(value) for key, value in data.items()}


def _record_values(record_cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    """Project an event payload onto a record's fields.

    Emitters attach correlation keys the record does not model (``call_id``,
    ``step_id``, ``truncated``, ``native_*``…); persisting must not blow up on
    them, and silently dropping them keeps the event stream the authoritative
    correlation source (the record is a metrics row, not a join table).
    """
    valid = {field.name for field in fields(record_cls)}
    values = {key: value for key, value in payload.items() if key in valid}
    for key in ("started_at", "finished_at"):
        if isinstance(values.get(key), str):
            values[key] = datetime.fromisoformat(values[key])
    return values


@dataclass(slots=True)
class LLMCallRecord:
    node_name: str
    provider: str
    model: str
    # None means the provider did not report the value. Zero is a real measured
    # value and must not be used as a substitute for unknown evaluation data.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return _payload(asdict(self))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LLMCallRecord":
        return cls(**_record_values(cls, payload))


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    tool_id: str | None = None
    server_name: str = "gateway"
    base_url: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    arguments: dict[str, Any] | None = None
    success: bool | None = None
    output: Any = None
    error: str | None = None
    risk_level: str | None = None
    risk_reason: str | None = None
    attempt: int = 1
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def to_payload(self) -> dict[str, Any]:
        return _payload(asdict(self))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ToolCallRecord":
        return cls(**_record_values(cls, payload))


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

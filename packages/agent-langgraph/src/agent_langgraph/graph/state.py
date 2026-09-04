from __future__ import annotations

import operator
from typing import Annotated

from common.enums import RunStatus, TaskStatus, VerificationResult, WorkflowPhase
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PlanStep(BaseModel):
    """
    One executable step produced by the planner.
    """

    id: int = Field(
        description="Sequential step number."
    )

    description: str = Field(
        description="Human-readable description of what must be done."
    )

    tool_name: str | None = Field(
        default=None,
        description="Preferred tool to execute this step."
    )

    arguments: dict = Field(
        default_factory=dict,
        description="Arguments for the selected tool."
    )

    verified: bool = Field(
        default=False,
        description="Whether the step has been verified."
    )

    verification: VerificationResult | None = Field(
        default=None,
        description="Result of the verification process."
    )

    status: RunStatus = RunStatus.PENDING

    output: str | None = None

class AgentPlan(BaseModel):
    """
    Structured planner output.
    """

    summary: str = Field(
        description="Short summary of the overall strategy."
    )

    reasoning: str = Field(
        description="Why this plan was chosen."
    )

    steps: list[PlanStep] = Field(
        description="Ordered executable steps."
    )

    requires_remediation: bool = Field(
        default=False,
        description=(
            "True only if this plan merely INVESTIGATES and its findings will "
            "need a follow-up plan that acts on them (e.g. 'find and fix bugs'). "
            "False when this plan already fully satisfies the goal."
        )
    )

class Finding(BaseModel):
    """
    One durable observation produced during an investigation phase.

    Findings outlive the plan that produced them: a replan replaces ``plan``
    wholesale, so anything the next phase needs must be lifted out of the step
    outputs and kept here.
    """

    step_id: int = Field(
        description="Plan step this finding came from."
    )

    description: str = Field(
        description="What was being investigated."
    )

    detail: str = Field(
        description="What was observed — the step output that matters."
    )

    source_tool: str | None = Field(
        default=None,
        description="Tool that produced the observation, if any."
    )

    phase: WorkflowPhase = Field(
        default=WorkflowPhase.INVESTIGATE,
        description="Phase during which the finding was recorded."
    )


class AgentState(TypedDict):
    """
    Checkpointed LangGraph execution state.

    Every node receives this state and returns only
    the fields it modifies.
    """

    # Conversation
    messages: Annotated[list[AnyMessage], add_messages]

    # Original request
    goal: str

    # Planner
    plan: AgentPlan

    current_step: int

    # Multi-phase workflow
    workflow_phase: WorkflowPhase | None

    # Accumulated across phases: appended to, never replaced, so investigation
    # results survive the replan that swaps out `plan`.
    findings: Annotated[list[Finding], operator.add]

    # Verification (verdict of the most recent verifier run)
    verification_success: bool | None

    verification_reason: str | None

    # Retry / Recovery
    retry_count: int

    last_error: str | None

    # Overall execution
    status: TaskStatus | None

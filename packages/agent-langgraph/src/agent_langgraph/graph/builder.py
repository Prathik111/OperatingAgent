from agent_langgraph.nodes.error_handler import ErrorHandlerNode
from agent_langgraph.nodes.executor import ExecutorNode
from agent_langgraph.nodes.phase_transition import PhaseTransitionNode
from agent_langgraph.nodes.planner import PlannerNode
from agent_langgraph.nodes.responder import ResponderNode
from agent_langgraph.nodes.verifier import VerifierNode
from langgraph.graph import END, START, StateGraph

from ..runtime.context import AgentContext
from .constants import (
    ERROR_HANDLER,
    EXECUTOR,
    PHASE_TRANSITION,
    PLANNER,
    RESPONDER,
    VERIFIER,
)
from .routing import phase_router, retry_router, should_execute, verification_router
from .state import AgentState


class GraphFactory:
    def create_graph(self) -> StateGraph[AgentState, AgentContext, AgentState, AgentState]:
        """
        Create a StateGraph based on the provided plan.
        """
        # Implementation to create a StateGraph from the plan

        Agent = StateGraph(AgentState, context_schema=AgentContext)

        Agent.add_node(PLANNER, PlannerNode)
        Agent.add_node(EXECUTOR, ExecutorNode)
        Agent.add_node(VERIFIER, VerifierNode)
        Agent.add_node(RESPONDER, ResponderNode)
        Agent.add_node(ERROR_HANDLER, ErrorHandlerNode)
        Agent.add_node(PHASE_TRANSITION, PhaseTransitionNode)

        Agent.add_edge(START, PLANNER)
        Agent.add_conditional_edges(
            PLANNER,
            should_execute,
            {
                EXECUTOR: EXECUTOR,
                RESPONDER: RESPONDER,
            },
        )
        Agent.add_edge(EXECUTOR, VERIFIER)
        Agent.add_conditional_edges(
            VERIFIER,
            verification_router,
            {
                EXECUTOR: EXECUTOR,
                PHASE_TRANSITION: PHASE_TRANSITION,
                ERROR_HANDLER: ERROR_HANDLER,
            },
        )
        Agent.add_conditional_edges(
            ERROR_HANDLER,
            retry_router,
            {
                PLANNER: PLANNER,
                RESPONDER: RESPONDER,
            },
        )
        Agent.add_conditional_edges(
            PHASE_TRANSITION,
            phase_router,
            {
                PLANNER: PLANNER,
                RESPONDER: RESPONDER,
            },
        )
        Agent.add_edge(RESPONDER, END)

        return Agent

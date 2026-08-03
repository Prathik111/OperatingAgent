class AgentException(Exception):
    pass


class PlanningException(AgentException):
    pass


class ToolExecutionException(AgentException):
    pass


class VerificationException(AgentException):
    pass
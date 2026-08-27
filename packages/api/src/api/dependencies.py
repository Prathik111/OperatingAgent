"""FastAPI dependency accessors that read the app state built by the lifespan.

Keeping these as thin ``Depends`` targets is what lets the unit tests override
them (via ``app.dependency_overrides``) and drive the routers through
``ASGITransport`` **without** running the lifespan — so the real repository,
broker and (eagerly-constructed) LangGraph orchestrator are never built.
"""

from __future__ import annotations

from fastapi import Request

from .config import ApiSettings
from .services.approval_gateway import ApprovalGateway
from .services.event_broker import EventBroker
from .services.task_service import TaskService


def get_settings(request: Request) -> ApiSettings:
    return request.app.state.settings


def get_task_service(request: Request) -> TaskService:
    return request.app.state.task_service


def get_broker(request: Request) -> EventBroker:
    return request.app.state.broker


def get_approval_gateway(request: Request) -> ApprovalGateway:
    return request.app.state.approvals

"""Low-level HTTP response hardening for the local API."""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityHeadersMiddleware:
    """Add browser-oriented security headers without buffering streamed bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                security_headers = {
                    "Cache-Control": "no-store",
                    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
                    "Referrer-Policy": "no-referrer",
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                }
                for name, value in security_headers.items():
                    if name not in headers:
                        headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_security_headers)

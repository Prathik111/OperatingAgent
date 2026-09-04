"""Tauri-local HTTP boundary security without application authentication."""

from __future__ import annotations


async def test_security_headers_are_added_without_buffering_response(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


async def test_tauri_origin_is_allowed(client):
    response = await client.options(
        "/tasks",
        headers={
            "Origin": "tauri://localhost",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "tauri://localhost"


async def test_untrusted_browser_origin_is_rejected(client):
    response = await client.options(
        "/tasks",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


async def test_untrusted_host_header_is_rejected(client):
    response = await client.get("/health", headers={"Host": "untrusted.example"})

    assert response.status_code == 400

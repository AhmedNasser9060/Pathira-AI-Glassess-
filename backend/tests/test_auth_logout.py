import pytest
from httpx import AsyncClient


async def _signup(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "Logout", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201
    return r.json()


@pytest.mark.asyncio
async def test_logout_revokes_refresh(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/auth/logout",
        json={"refresh_token": body["refresh_token"]},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    # Refresh should now fail
    r2 = await client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": body["refresh_token"]},
    )
    assert r2.status_code == 401


@pytest.mark.asyncio
async def test_logout_requires_auth_header(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/auth/logout",
        json={"refresh_token": body["refresh_token"]},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_logout_idempotent_on_unknown_token(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/auth/logout",
        json={"refresh_token": "garbage"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    # Should still 200 — logout is idempotent; we don't leak which tokens existed.
    assert r.status_code == 200
